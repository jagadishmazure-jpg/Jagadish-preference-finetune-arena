"""A small decoder-only transformer: embeddings, causal multi-head attention, RoPE, and a KV cache.

Sampling supports temperature, top-k and top-p. Training is next-token cross-entropy on a token stream.
No library transformer block is used. The CPU smoke is a few dozen steps; the Azure ML job is
llm-foundations/azure/minigpt-job.yml and has not been run.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch import nn

from .tokenizer import ByteBPE

CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "fenwold.txt"


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x is (batch, heads, time, dim). cos and sin are (time, dim)."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


def rope_tables(head_dim: int, block_size: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2:
        raise ValueError("RoPE needs an even head dimension")
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(block_size).float(), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def sample_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """One token per row. temperature <= 0 or top_k == 1 is greedy."""
    if temperature <= 0 or top_k == 1:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k and top_k < logits.size(-1):
        cutoff = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    if top_p < 1.0:
        ordered, index = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(ordered, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        ordered = ordered.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, index, ordered)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        if n_embd % n_head:
            raise ValueError("n_embd must divide into n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        cos, sin = rope_tables(self.head_dim, block_size)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor] | None = None,
                return_attn: bool = False):
        b, t, _c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (part.transpose(1, 2) for part in (q, k, v))  # (B, H, T, D)
        past = 0 if cache is None else cache[0].shape[2]
        if past + t > self.cos.shape[0]:
            raise ValueError("sequence is longer than the RoPE table")
        cos, sin = self.cos[past: past + t], self.sin[past: past + t]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        key_pos = torch.arange(k.shape[2], device=x.device)
        query_pos = torch.arange(past, past + t, device=x.device)
        causal = key_pos.view(1, 1, 1, -1) > query_pos.view(1, 1, -1, 1)
        scores = scores.masked_fill(causal, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        y = (weights @ v).transpose(1, 2).contiguous().view(b, t, -1)
        return self.proj(y), (k, v), weights if return_attn else None


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x, cache=None, return_attn=False):
        y, cache, weights = self.attn(self.ln1(x), cache, return_attn)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, cache, weights


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int = 32, n_head: int = 4, n_layer: int = 2, block_size: int = 64):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList(Block(n_embd, n_head, block_size) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, cache: list | None = None, return_attn: bool = False):
        x = self.tok_emb(idx)
        new_cache, attns = [], []
        for i, block in enumerate(self.blocks):
            past = None if cache is None else cache[i]
            x, past, weights = block(x, past, return_attn)
            new_cache.append(past)
            if return_attn:
                attns.append(weights)
        return self.lm_head(self.ln_f(x)), new_cache, attns

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int, temperature: float = 1.0, top_k: int = 0,
                 top_p: float = 1.0, use_cache: bool = True, generator: torch.Generator | None = None) -> torch.Tensor:
        self.eval()
        cache = None
        for _ in range(max_new):
            if idx.size(1) >= self.block_size:
                break
            if use_cache and cache is not None:
                logits, cache, _ = self.forward(idx[:, -1:], cache=cache)
            else:
                logits, full, _ = self.forward(idx, cache=None)
                cache = full if use_cache else None
            nxt = sample_token(logits[:, -1, :], temperature, top_k, top_p, generator)
            idx = torch.cat((idx, nxt[:, None]), dim=1)
        return idx


def load_corpus(path: Path = CORPUS) -> str:
    return path.read_text(encoding="utf-8")


def prepare(num_merges: int = 20, repeats: int = 4) -> tuple[ByteBPE, torch.Tensor]:
    text = load_corpus() * repeats
    tok = ByteBPE().train(text, num_merges)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    return tok, ids


def train_steps(model: MiniGPT, ids: torch.Tensor, steps: int, batch: int, block: int, lr: float, seed: int,
                device: torch.device | str | None = None) -> list[float]:
    if device is not None:
        model.to(device)
        ids = ids.to(device)
    generator = torch.Generator()
    generator.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    span = len(ids) - block - 1
    if span < 1:
        raise ValueError("corpus is shorter than the block")
    for _ in range(steps):
        starts = torch.randint(0, span, (batch,), generator=generator)
        x = torch.stack([ids[i: i + block] for i in starts.tolist()])
        y = torch.stack([ids[i + 1: i + 1 + block] for i in starts.tolist()])
        logits, _, _ = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(round(float(loss.detach()), 4))
    model.eval()
    return losses


def save_checkpoint(path: Path, model: MiniGPT, tok: ByteBPE, config: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "merges": tok.merges,
        "config": config,
    }, path / "minigpt.pt")


def attention_svg(weights: torch.Tensor, tokens: list[str]) -> str:
    """One attention head as an SVG heatmap. weights is (time, time) and causal cells can be zero."""
    grid = weights.detach().float().cpu()
    t = grid.shape[0]
    if len(tokens) != t:
        raise ValueError("one token label per row")
    cell, label = 16, 72
    size = label + cell * t

    def esc(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" font-family="sans-serif" font-size="9">']
    peak = float(grid.max().clamp(min=1e-8))
    for i in range(t):
        parts.append(f'<text x="2" y="{label + i * cell + 12}">{esc(tokens[i][:8])}</text>')
        for j in range(t):
            shade = int(255 - 255 * float(grid[i, j]) / peak)
            parts.append(
                f'<rect x="{label + j * cell}" y="{label + i * cell}" width="{cell - 1}" height="{cell - 1}" '
                f'fill="rgb({shade},{shade},255)"/>'
            )
    parts.append("</svg>")
    return "".join(parts)


@torch.no_grad()
def benchmark_kv(model: MiniGPT, prompt: torch.Tensor, new_tokens: int) -> dict:
    """Greedy decode with and without the KV cache. Tokens must match; times are informational."""
    def timed(use_cache: bool):
        start = time.perf_counter()
        out = model.generate(prompt.clone(), new_tokens, temperature=0.0, use_cache=use_cache)
        return out, time.perf_counter() - start

    full, full_s = timed(False)
    cached, cache_s = timed(True)
    return {
        "match": torch.equal(full, cached),
        "full_s": full_s,
        "cache_s": cache_s,
        "new_tokens": int(full.size(1) - prompt.size(1)),
    }


def smoke(steps: int = 20) -> dict:
    """A few CPU steps on the Fenwold corpus. Returns flags, not timings."""
    tok, ids = prepare(num_merges=16, repeats=2)
    model = MiniGPT(tok.vocab_size, n_embd=32, n_head=4, n_layer=2, block_size=48)
    losses = train_steps(model, ids, steps=steps, batch=8, block=24, lr=3e-3, seed=0)
    prompt = ids[:8].view(1, -1)
    bench = benchmark_kv(model, prompt, new_tokens=8)
    return {
        "loss_fell": losses[-1] < losses[0],
        "kv_match": bench["match"],
        "steps": len(losses),
        "vocab": tok.vocab_size,
    }
