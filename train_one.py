from __future__ import annotations

from torch.profiler import profile, ProfilerActivity
import os, math, gzip, argparse, random, json, csv, time, inspect
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint

from transformers import AutoTokenizer
from tqdm import tqdm

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

TOKENIZER_ID = "speakleash/Bielik-7B-v0.1"


FLASH_AVAILABLE = False
flash_attn_func = None
try:
    from flash_attn import flash_attn_func as _flash_attn_func
    flash_attn_func = _flash_attn_func
    FLASH_AVAILABLE = True
except Exception:
    FLASH_AVAILABLE = False
    flash_attn_func = None


def mb(x_bytes: int) -> float:
    return x_bytes / (1024 ** 2)

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def tok_encode(tokenizer, text: str):
    return tokenizer.encode(text, add_special_tokens=False)

def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
    return tok, tok.pad_token_id, tok.bos_token_id, tok.eos_token_id

def _read_text(fp: Path) -> str:
    if str(fp).endswith(".gz"):
        with gzip.open(fp, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return fp.read_text(encoding="utf-8", errors="ignore")

def encode_corpus(tokenizer, files, bos_id=None, eos_id=None):
    ids: List[int] = []
    for fp in files:
        txt = _read_text(fp)
        for ln in txt.splitlines():
            if not ln.strip():
                continue
            piece = tok_encode(tokenizer, ln)
            if bos_id is not None:
                piece = [bos_id] + piece
            if eos_id is not None:
                piece = piece + [eos_id]
            ids.extend(piece)
    return ids

def log_metrics_csv(path: Path, row: dict, field_order):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        if new:
            w.writeheader()
        w.writerow(row)

def is_oom(e: Exception) -> bool:
    msg = str(e).lower()
    return ("out of memory" in msg) or ("cuda oom" in msg)

def reset_cuda_peak():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cuda_sync()


def profile_memory_with_torch_profiler(
    model,
    loader,
    pad_id,
    amp_bf16: bool,
    warmup_steps=20,
    profile_steps=5,
) -> Dict[str, Any]:
    assert torch.cuda.is_available(), "Profil pamięci sensowny tylko na CUDA"
    device = next(model.parameters()).device
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    it = iter(loader)

    def step_one_warmup(batch):
        x = batch["input_ids"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=amp_bf16):
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=pad_id if pad_id is not None else -100,
            )
        loss.backward()
        opt.step()

    def step_one_measure(batch) -> Dict[str, float]:
        """
        Zwraca peak pamięci (allocated/reserved) osobno dla:
        - forward+loss
        - backward
        Uwaga: wartości są w MB.
        """
        x = batch["input_ids"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        cuda_sync()

        # reset peak, ale baza to aktualnie zaalokowana pamięć (parametry, cache itd.)
        base_alloc_f = torch.cuda.memory_allocated()
        base_res_f   = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        cuda_sync()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=amp_bf16):
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=pad_id if pad_id is not None else -100,
            )

        cuda_sync()
        f_peak_alloc = torch.cuda.max_memory_allocated()
        f_peak_res   = torch.cuda.max_memory_reserved()

        # -------- backward --------
        base_alloc_b = torch.cuda.memory_allocated()
        base_res_b   = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        cuda_sync()

        loss.backward()

        cuda_sync()
        b_peak_alloc = torch.cuda.max_memory_allocated()
        b_peak_res   = torch.cuda.max_memory_reserved()

        # opt.step nie wchodzi do backward-mem
        opt.step()

        return {
            # absolutne piki (w praktyce: peak od aktualnej bazy)
            "forward_peak_alloc_mb":  round(mb(f_peak_alloc), 2),
            "forward_peak_res_mb":    round(mb(f_peak_res), 2),
            "backward_peak_alloc_mb": round(mb(b_peak_alloc), 2),
            "backward_peak_res_mb":   round(mb(b_peak_res), 2),

            # opcjonalnie: przyrosty względem bazy (często wygodniejsze do porównań)
            "forward_delta_alloc_mb":  round(mb(max(0, f_peak_alloc - base_alloc_f)), 2),
            "forward_delta_res_mb":    round(mb(max(0, f_peak_res   - base_res_f)), 2),
            "backward_delta_alloc_mb": round(mb(max(0, b_peak_alloc - base_alloc_b)), 2),
            "backward_delta_res_mb":   round(mb(max(0, b_peak_res   - base_res_b)), 2),
        }

    # warmup
    for _ in range(warmup_steps):
        batch = _next_batch(it)
        if batch is None:
            it = iter(loader)
            batch = next(it)
        step_one_warmup(batch)

    reset_cuda_peak()

    # agregacja forward/backward po kilku krokach (bierzemy MAX)
    f_alloc_max = f_res_max = 0.0
    b_alloc_max = b_res_max = 0.0
    f_dalloc_max = f_dres_max = 0.0
    b_dalloc_max = b_dres_max = 0.0

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, profile_memory=True, record_shapes=False) as prof:
        for _ in range(profile_steps):
            batch = _next_batch(it)
            if batch is None:
                it = iter(loader)
                batch = next(it)

            m = step_one_measure(batch)

            f_alloc_max = max(f_alloc_max, m["forward_peak_alloc_mb"])
            f_res_max   = max(f_res_max,   m["forward_peak_res_mb"])
            b_alloc_max = max(b_alloc_max, m["backward_peak_alloc_mb"])
            b_res_max   = max(b_res_max,   m["backward_peak_res_mb"])

            f_dalloc_max = max(f_dalloc_max, m["forward_delta_alloc_mb"])
            f_dres_max   = max(f_dres_max,   m["forward_delta_res_mb"])
            b_dalloc_max = max(b_dalloc_max, m["backward_delta_alloc_mb"])
            b_dres_max   = max(b_dres_max,   m["backward_delta_res_mb"])

            prof.step()

    cuda_sync()
    peak_alloc_mb = round(mb(torch.cuda.max_memory_allocated()), 2)
    peak_reserved_mb = round(mb(torch.cuda.max_memory_reserved()), 2)

    return {
        "warmup_steps": warmup_steps,
        "profile_steps": profile_steps,

        # Twoje dotychczasowe "peak" dla całego okna profilowania
        "peak_allocated_mb": peak_alloc_mb,
        "peak_reserved_mb": peak_reserved_mb,

        # NOWE: forward/backward (MAX z kilku kroków)
        "forward_peak_alloc_mb": f_alloc_max,
        "forward_peak_reserved_mb": f_res_max,
        "backward_peak_alloc_mb": b_alloc_max,
        "backward_peak_reserved_mb": b_res_max,

        # opcjonalnie (często lepsze do raportu): przyrosty pamięci
        "forward_delta_alloc_mb": f_dalloc_max,
        "forward_delta_reserved_mb": f_dres_max,
        "backward_delta_alloc_mb": b_dalloc_max,
        "backward_delta_reserved_mb": b_dres_max,

        "top_ops_by_self_cuda_memory": prof.key_averages().table(
            sort_by="self_cuda_memory_usage", row_limit=15
        )
    }




class CLMDataset(Dataset):
    def __init__(self, token_ids: List[int], seq_len=128, stride: Optional[int] = None):
        self.ids = token_ids
        self.seq_len = seq_len
        self.stride = stride or seq_len

        if len(self.ids) < (self.seq_len + 1):
            self.n = 0
        else:
            avail = len(self.ids) - (self.seq_len + 1)
            self.n = avail // self.stride + 1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        start = i * self.stride
        end = start + self.seq_len + 1
        chunk = self.ids[start:end]
        if len(chunk) < self.seq_len + 1:
            start = max(0, len(self.ids) - (self.seq_len + 1))
            chunk = self.ids[start:start + self.seq_len + 1]

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return {"input_ids": x, "labels": y}

def collate(batch):
    x = torch.stack([b["input_ids"] for b in batch])
    y = torch.stack([b["labels"]    for b in batch])
    return {"input_ids": x, "labels": y}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=4096):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        L = x.size(1)
        x = x + self.pe[:L].unsqueeze(0)
        return self.drop(x)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, padding_idx=None):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class MultiHeadedAttention(nn.Module):
    """
    - baseline: SDPA (PyTorch)
    - optional: flash-attn (causal), optional window_size (local attention)
    """
    def __init__(self, h, d_model, dropout=0.1, use_flash_attn=False, window_size: Optional[int]=None):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.dk = d_model // h
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop_p = dropout

        self.use_flash_attn = use_flash_attn
        self.window_size = window_size  # if set => local attention (requires flash-attn)

    def _split(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.h, self.dk)

    def _merge(self, x):
        B, L, H, dk = x.shape
        return x.view(B, L, H * dk)

    def forward(self, x):
        # x: [B, L, D]
        q = self._split(self.q(x))
        k = self._split(self.k(x))
        v = self._split(self.v(x))

        if self.use_flash_attn and FLASH_AVAILABLE:
            # flash-attn expects [B, L, H, dk]
            dropout_p = self.drop_p if self.training else 0.0

            # windowed causal attention: left=window, right=0
            ws = None
            if self.window_size is not None:
                ws = (int(self.window_size), 0)

            # different flash-attn versions have different signatures -> be defensive
            try:
                sig = inspect.signature(flash_attn_func)
                if "window_size" in sig.parameters:
                    y = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=True, window_size=ws)
                else:
                    # old versions: no window_size
                    if ws is not None:
                        # can't do windowed without window_size support
                        y = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=True)
                    else:
                        y = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=True)
            except TypeError:
                # fallback signature
                y = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=True)

            return self.o(self._merge(y))

        # SDPA fallback (causal)
        # convert to [B, H, L, dk]
        q2 = q.transpose(1, 2)
        k2 = k.transpose(1, 2)
        v2 = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q2, k2, v2,
            attn_mask=None,
            dropout_p=self.drop_p if self.training else 0.0,
            is_causal=True
        )  # [B,H,L,dk]
        y = y.transpose(1, 2).contiguous()  # [B,L,H,dk]
        return self.o(self._merge(y))

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.drop(F.gelu(self.w1(x))))

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, use_flash_attn=False, window_size: Optional[int]=None):
        super().__init__()
        self.self_attn = MultiHeadedAttention(n_heads, d_model, dropout, use_flash_attn=use_flash_attn, window_size=window_size)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.n1 = nn.LayerNorm(d_model, eps=1e-5)
        self.n2 = nn.LayerNorm(d_model, eps=1e-5)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.self_attn(self.n1(x)))
        x = x + self.drop(self.ffn(self.n2(x)))
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
        max_len=2048,
        pad_id=None,
        use_flash_attn: bool=False,
        window_size: Optional[int]=None,
        grad_ckpt: bool=False,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.grad_ckpt = grad_ckpt

        self.tok = Embeddings(d_model, vocab_size, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout, use_flash_attn=use_flash_attn, window_size=window_size)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.lm = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.pos(self.tok(x))
        if self.grad_ckpt and self.training:
            for blk in self.layers:
                h = checkpoint(blk, h, use_reentrant=False)
        else:
            for blk in self.layers:
                h = blk(h)
        h = self.norm(h)
        logits = self.lm(h)  # logits, NOT log_softmax
        return logits


@torch.no_grad()
def _next_batch(it):
    try:
        return next(it)
    except StopIteration:
        return None


def train_epoch(
    model,
    loader,
    opt,
    pad_id,
    amp_bf16: bool,
    accum_steps=1,
    update_every=20,
):
    time_hist: List[float] = []

    model.train()
    total_loss, total_tok = 0.0, 0
    opt.zero_grad(set_to_none=True)

    epoch_peak_alloc_mb = float("nan")
    epoch_peak_reserved_mb = float("nan")

    if torch.cuda.is_available():
        reset_cuda_peak()

    bar = tqdm(total=len(loader), desc="train", dynamic_ncols=True)

    for i, batch in enumerate(loader, 1):
        x = batch["input_ids"].to(next(model.parameters()).device, non_blocking=True)
        y = batch["labels"].to(x.device, non_blocking=True)

        if torch.cuda.is_available():
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=amp_bf16):
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=pad_id if pad_id is not None else -100,
            )
            loss = loss / accum_steps

        loss.backward()

        if torch.cuda.is_available():
            ender.record()
            cuda_sync()
            step_time_ms = starter.elapsed_time(ender)
            time_hist.append(step_time_ms)
            if len(time_hist) > 20:
                time_hist.pop(0)

        if i % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            n_tok = y.numel() if pad_id is None else (y != pad_id).sum().item()
            total_loss += (loss.item() * accum_steps) * n_tok
            total_tok += n_tok

        if i % update_every == 0:
            ppl = math.exp(total_loss / max(1, total_tok))
            if torch.cuda.is_available() and time_hist:
                bar.set_postfix_str(f"ppl~{ppl:.1f} | step_ms~{sum(time_hist)/len(time_hist):.1f}")
            else:
                bar.set_postfix_str(f"ppl~{ppl:.1f}")
        bar.update(1)

    bar.close()

    if torch.cuda.is_available():
        cuda_sync()
        epoch_peak_alloc_mb = mb(torch.cuda.max_memory_allocated())
        epoch_peak_reserved_mb = mb(torch.cuda.max_memory_reserved())

    step_time_ms_mean20 = (sum(time_hist) / len(time_hist)) if time_hist else float("nan")
    train_ppl = math.exp(total_loss / max(1, total_tok))
    return train_ppl, step_time_ms_mean20, epoch_peak_alloc_mb, epoch_peak_reserved_mb


@torch.no_grad()
def eval_ppl(model, loader, pad_id, amp_bf16: bool):
    model.eval()
    total_loss, total_tok = 0.0, 0

    for batch in tqdm(loader, desc="valid", leave=False, dynamic_ncols=True):
        x = batch["input_ids"].to(next(model.parameters()).device)
        y = batch["labels"].to(x.device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=amp_bf16):
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=pad_id if pad_id is not None else -100,
                reduction="sum",
            )

        total_loss += loss.item()
        total_tok += y.numel() if pad_id is None else (y != pad_id).sum().item()

    return math.exp(total_loss / max(1, total_tok))


def can_run_one_step(
    model_ctor,
    model_kwargs,
    ds,
    workers: int,
    batch_size: int,
    amp_bf16: bool,
) -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_ctor(**model_kwargs).to(device)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True,
        collate_fn=collate, num_workers=workers, pin_memory=True,
        persistent_workers=True if workers > 0 else False
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    batch = next(iter(dl))
    try:
        reset_cuda_peak()
        x = batch["input_ids"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=amp_bf16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=model_kwargs.get("pad_id", -100))
        loss.backward()
        opt.step()
        return True
    except Exception as e:
        if is_oom(e):
            return False
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", default="out_run")

    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--stride",  type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=1)

    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.02)

    # toggles for techniques
    ap.add_argument("--amp_bf16", action="store_true", help="1) BF16 AMP autocast")
    ap.add_argument("--use_flash_attn", action="store_true", help="2) FlashAttention (requires flash-attn + BF16)")
    ap.add_argument("--window_size", type=int, default=None, help="3) Local attention window (requires flash-attn). e.g. 256")
    ap.add_argument("--grad_ckpt", action="store_true", help="4) Gradient checkpointing")

    # profiling
    ap.add_argument("--profile_memory", action="store_true", help="torch.profiler memory check (warmup+few steps)")
    ap.add_argument("--profile_steps", type=int, default=5)
    ap.add_argument("--warmup_steps", type=int, default=20)

    # optional search for max batch
    ap.add_argument("--find_max_batch", action="store_true", help="Try to find max batch size that fits (quick 1-step test)")
    ap.add_argument("--max_batch_limit", type=int, default=4096)

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer, pad_id, bos_id, eos_id = load_tokenizer()

    vocab_size = tokenizer.vocab_size
    vocab_size = max(
        vocab_size,
        (pad_id or -1) + 1,
        (bos_id or -1) + 1,
        (eos_id or -1) + 1,
    )

    # enforce: flash-attn => BF16
    if args.use_flash_attn and not args.amp_bf16:
        print("[warn] --use_flash_attn wymaga BF16 AMP -> włączam --amp_bf16.")
        args.amp_bf16 = True

    # windowed => flash-attn
    if args.window_size is not None and not args.use_flash_attn:
        print("[warn] --window_size wymaga flash-attn -> włączam --use_flash_attn (i BF16).")
        args.use_flash_attn = True
        args.amp_bf16 = True

    if args.use_flash_attn and not FLASH_AVAILABLE:
        print("[warn] --use_flash_attn ustawione, ale flash-attn nie jest dostępne -> fallback SDPA.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_files = sorted(Path(args.data_dir).glob("*.txt")) + sorted(Path(args.data_dir).glob("*.txt.gz"))
    assert all_files, "Brak plików .txt/.txt.gz w data_dir"

    all_ids = encode_corpus(tokenizer, all_files, bos_id, eos_id)
    cut = int((1.0 - args.val_ratio) * len(all_ids))
    train_ids = all_ids[:cut]
    valid_ids = all_ids[cut:]

    train_ds = CLMDataset(train_ids, seq_len=args.seq_len, stride=args.stride or args.seq_len)
    valid_ds = CLMDataset(valid_ids, seq_len=args.seq_len, stride=args.stride or args.seq_len)

    # build model kwargs (for reuse in find_max_batch)
    model_kwargs = dict(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pad_id=pad_id,
        max_len=4096,
        use_flash_attn=args.use_flash_attn and FLASH_AVAILABLE,
        window_size=args.window_size if (args.use_flash_attn and FLASH_AVAILABLE) else None,
        grad_ckpt=args.grad_ckpt,
    )

    if args.find_max_batch and torch.cuda.is_available() and len(train_ds) > 0:
        print("[find_max_batch] szukam max batch dla tej konfiguracji (1 krok fw+bw)...")
        # exponential search then binary
        lo = 1
        hi = 1
        while hi <= args.max_batch_limit and can_run_one_step(DecoderOnlyTransformer, model_kwargs, train_ds, args.workers, hi, args.amp_bf16):
            lo = hi
            hi *= 2
        hi = min(hi, args.max_batch_limit)
        # binary between lo..hi (hi might fail)
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            ok = can_run_one_step(DecoderOnlyTransformer, model_kwargs, train_ds, args.workers, mid, args.amp_bf16)
            if ok:
                lo = mid
            else:
                hi = mid
        print(f"[find_max_batch] max_batch ~= {lo}")
        # you can choose to override:
        args.batch_size = lo

    model = DecoderOnlyTransformer(**model_kwargs).to(device)

    dl_train = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=collate, num_workers=args.workers, pin_memory=True,
        prefetch_factor=4 if args.workers > 0 else None,
        persistent_workers=True if args.workers > 0 else False
    )
    dl_valid = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        collate_fn=collate, num_workers=max(1, args.workers // 2), pin_memory=True
    )

    # CSV logging
    log_path = Path(args.output_dir) / "train_log.csv"
    log_fields = [
        "epoch", "split", "loss", "ppl",
        "epoch_time_s", "elapsed_s", "tokens_per_s",
        "batch_size", "seq_len",
        "step_ms_mean20",
        "epoch_peak_alloc_mb", "epoch_peak_reserved_mb",
        "memprof_peak_alloc_mb", "memprof_peak_reserved_mb",

        # NEW:
        "memprof_forward_peak_alloc_mb", "memprof_forward_peak_reserved_mb",
        "memprof_backward_peak_alloc_mb", "memprof_backward_peak_reserved_mb",
        "memprof_forward_delta_alloc_mb", "memprof_forward_delta_reserved_mb",
        "memprof_backward_delta_alloc_mb", "memprof_backward_delta_reserved_mb",

        "tech",
        "flash_attn", "bf16", "window_size", "grad_ckpt",
    ]

    def tech_name():
        if args.use_flash_attn and args.window_size is not None:
            return "3_local_flashattn_bf16"
        if args.use_flash_attn:
            return "2_flashattn_bf16"
        if args.grad_ckpt:
            return "4_grad_ckpt" + ("_bf16" if args.amp_bf16 else "_fp32")
        if args.amp_bf16:
            return "1_amp_bf16"
        return "0_baseline_fp32"

    run_start = time.time()

    # optional torch.profiler memory check
    memprof_payload = None
    if args.profile_memory and torch.cuda.is_available() and len(train_ds) > 0:
        try:
            memprof_info = profile_memory_with_torch_profiler(
                model=model,
                loader=dl_train,
                pad_id=pad_id,
                amp_bf16=args.amp_bf16,
                warmup_steps=args.warmup_steps,
                profile_steps=args.profile_steps,
            )
            memprof_payload = memprof_info

            prof_path = Path(args.output_dir) / "memory_profile.json"
            prof_path.write_text(json.dumps({
                "tech": tech_name(),
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                **memprof_info,
            }, indent=2, ensure_ascii=False), encoding="utf-8")

            print("\n[memory_profiler] Saved:", prof_path)
            print("[memory_profiler] peak_allocated_mb:", memprof_info["peak_allocated_mb"])
            print("[memory_profiler] peak_reserved_mb:", memprof_info["peak_reserved_mb"])

            log_metrics_csv(log_path, {
                "epoch": 0,
                "split": "memprof",
                "loss": "",
                "ppl": "",
                "epoch_time_s": "",
                "elapsed_s": round(time.time() - run_start, 3),
                "tokens_per_s": "",
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "step_ms_mean20": "",
                "epoch_peak_alloc_mb": "",
                "epoch_peak_reserved_mb": "",
                "memprof_peak_alloc_mb": memprof_info["peak_allocated_mb"],
                "memprof_peak_reserved_mb": memprof_info["peak_reserved_mb"],
                "tech": tech_name(),
                "flash_attn": bool(args.use_flash_attn and FLASH_AVAILABLE),
                "bf16": bool(args.amp_bf16),
                "window_size": args.window_size if args.window_size is not None else "",
                "grad_ckpt": bool(args.grad_ckpt),

                # NEW:
                "memprof_forward_peak_alloc_mb": memprof_info.get("forward_peak_alloc_mb", ""),
                "memprof_forward_peak_reserved_mb": memprof_info.get("forward_peak_reserved_mb", ""),
                "memprof_backward_peak_alloc_mb": memprof_info.get("backward_peak_alloc_mb", ""),
                "memprof_backward_peak_reserved_mb": memprof_info.get("backward_peak_reserved_mb", ""),
                
                "memprof_forward_delta_alloc_mb": memprof_info.get("forward_delta_alloc_mb", ""),
                "memprof_forward_delta_reserved_mb": memprof_info.get("forward_delta_reserved_mb", ""),
                "memprof_backward_delta_alloc_mb": memprof_info.get("backward_delta_alloc_mb", ""),
                "memprof_backward_delta_reserved_mb": memprof_info.get("backward_delta_reserved_mb", ""),



            }, field_order=log_fields)

        except Exception as e:
            if is_oom(e):
                print("[warn] OOM w torch.profiler. To normalne przy dużym bs. "
                      "Uruchom profilowanie z mniejszym --batch_size albo wyłącz --profile_memory.")
            else:
                raise

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    best = float("inf")
    for e in range(1, args.epochs + 1):
        epoch_start = time.time()

        tr_ppl, step_ms_mean20, epoch_peak_alloc_mb, epoch_peak_reserved_mb = train_epoch(
            model, dl_train, opt, pad_id,
            amp_bf16=args.amp_bf16,
            accum_steps=args.accum_steps
        )
        va_ppl = eval_ppl(model, dl_valid, pad_id, amp_bf16=args.amp_bf16) if len(valid_ds) > 0 else float("nan")

        epoch_sec = time.time() - epoch_start
        elapsed_sec = time.time() - run_start
        steps = len(dl_train)
        tokens_epoch_est = steps * args.batch_size * args.seq_len
        tps = tokens_epoch_est / max(1e-9, epoch_sec)

        tr_loss = math.log(tr_ppl) if tr_ppl > 0 else float("nan")
        va_loss = math.log(va_ppl) if (va_ppl > 0 and not math.isnan(va_ppl)) else float("nan")

        print(
            f"[epoch {e}] train PPL={tr_ppl:.2f} | valid PPL={va_ppl:.2f} | "
            f"epoch={epoch_sec:6.1f}s | tok/s~{tps:,.0f} | bs={args.batch_size} | "
            f"epoch_peak_alloc={epoch_peak_alloc_mb:.0f}MB | flash_attn={bool(args.use_flash_attn and FLASH_AVAILABLE)} | bf16={bool(args.amp_bf16)}"
        )

        log_metrics_csv(log_path, {
            "epoch": e,
            "split": "train",
            "loss": tr_loss,
            "ppl": tr_ppl,
            "epoch_time_s": round(epoch_sec, 3),
            "elapsed_s": round(elapsed_sec, 3),
            "tokens_per_s": round(tps, 1),
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "step_ms_mean20": round(step_ms_mean20, 3) if not math.isnan(step_ms_mean20) else "",
            "epoch_peak_alloc_mb": round(epoch_peak_alloc_mb, 1) if not math.isnan(epoch_peak_alloc_mb) else "",
            "epoch_peak_reserved_mb": round(epoch_peak_reserved_mb, 1) if not math.isnan(epoch_peak_reserved_mb) else "",
            "memprof_peak_alloc_mb": memprof_payload["peak_allocated_mb"] if memprof_payload else "",
            "memprof_peak_reserved_mb": memprof_payload["peak_reserved_mb"] if memprof_payload else "",
            "tech": tech_name(),
            "flash_attn": bool(args.use_flash_attn and FLASH_AVAILABLE),
            "bf16": bool(args.amp_bf16),
            "window_size": args.window_size if args.window_size is not None else "",
            "grad_ckpt": bool(args.grad_ckpt),
        }, field_order=log_fields)

        if not math.isnan(va_ppl):
            log_metrics_csv(log_path, {
                "epoch": e,
                "split": "valid",
                "loss": va_loss,
                "ppl": va_ppl,
                "epoch_time_s": round(epoch_sec, 3),
                "elapsed_s": round(elapsed_sec, 3),
                "tokens_per_s": round(tps, 1),
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "step_ms_mean20": "",
                "epoch_peak_alloc_mb": "",
                "epoch_peak_reserved_mb": "",
                "memprof_peak_alloc_mb": "",
                "memprof_peak_reserved_mb": "",
                "tech": tech_name(),
                "flash_attn": bool(args.use_flash_attn and FLASH_AVAILABLE),
                "bf16": bool(args.amp_bf16),
                "window_size": args.window_size if args.window_size is not None else "",
                "grad_ckpt": bool(args.grad_ckpt),
            }, field_order=log_fields)

        torch.save(model.state_dict(), Path(args.output_dir) / "last.pt")
        if va_ppl < best:
            best = va_ppl
            torch.save(model.state_dict(), Path(args.output_dir) / "best.pt")


if __name__ == "__main__":
    main()
