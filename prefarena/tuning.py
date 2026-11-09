"""SFT, DPO and ORPO with LoRA on a tiny open model, on CPU. Needs torch and transformers.

Everything here is small on purpose: a 135M-parameter model, about a million LoRA parameters,
responses capped at 128 tokens, and a few hundred pairs. Reference log-probs for DPO are computed once
with the adapters switched off, so only one copy of the model is in memory.
"""

from __future__ import annotations

import json
import random
import time

import torch

from .lora import apply_lora, merge_lora, set_adapter
from .losses import dpo_loss, orpo_loss
from .styles import messages

MAX_RESPONSE_TOKENS = 128


def load(path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    model.eval()
    return tok, model


def encode(tok, p: dict, system: str, response: str | None = None) -> tuple[list[int], list[int]]:
    prompt = tok.apply_chat_template(messages(p, system), add_generation_prompt=True, tokenize=True)
    if isinstance(prompt, dict) or hasattr(prompt, "input_ids"):
        prompt = prompt["input_ids"]
    if response is None:
        return list(prompt), []
    resp = tok(response, add_special_tokens=False)["input_ids"][:MAX_RESPONSE_TOKENS] + [tok.eos_token_id]
    return list(prompt), resp


def seq_logps(model, batch: list[tuple[list[int], list[int]]], pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed and mean log-prob of each response given its prompt (right padding)."""
    seqs = [p + r for p, r in batch]
    n = max(map(len, seqs))
    ids = torch.full((len(seqs), n), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), n), dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.float32)
    for i, ((p, _r), s) in enumerate(zip(batch, seqs)):
        ids[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = 1
        mask[i, len(p): len(s)] = 1
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    m = mask[:, 1:]
    total = (lp * m).sum(-1)
    return total, total / m.sum(-1).clamp(min=1)


def _batches(items: list, bs: int, seed: int, epoch: int):
    order = list(range(len(items)))
    random.Random(f"{seed}-{epoch}").shuffle(order)
    for i in range(0, len(order), bs):
        yield [items[j] for j in order[i: i + bs]]


def _opt(model, lr: float):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)


def train_sft(model, tok, examples: list[tuple], cfg: dict, log: list) -> None:
    """examples: (prompt_ids, response_ids). Mean token NLL of the response."""
    torch.manual_seed(cfg["seed"])
    apply_lora(model, r=cfg["lora_r"], alpha=cfg["lora_alpha"])
    opt, step = _opt(model, cfg["lr"]), 0
    model.train()
    for ep in range(cfg["epochs"]):
        for b in _batches(examples, cfg["batch"], cfg["seed"], ep):
            t0 = time.perf_counter()
            _, mean = seq_logps(model, b, tok.pad_token_id)
            loss = -mean.mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            log.append({"stage": "sft", "epoch": ep, "step": step, "loss": round(loss.item(), 4),
                        "seconds": round(time.perf_counter() - t0, 3)})
    model.eval()


def train_pref(model, tok, pairs: list[tuple], cfg: dict, log: list, method: str) -> None:
    """pairs: (prompt_ids, chosen_ids, rejected_ids). method: dpo or orpo."""
    torch.manual_seed(cfg["seed"])
    apply_lora(model, r=cfg["lora_r"], alpha=cfg["lora_alpha"])
    ref = None
    if method == "dpo":  # reference = current weights with adapters off (B = 0 makes them equal at start)
        set_adapter(model, False)
        with torch.no_grad():
            ref = []
            for i in range(0, len(pairs), 8):
                chunk = pairs[i: i + 8]
                c, _ = seq_logps(model, [(p, ch) for p, ch, _ in chunk], tok.pad_token_id)
                r, _ = seq_logps(model, [(p, rj) for p, _, rj in chunk], tok.pad_token_id)
                ref += list(zip(c.tolist(), r.tolist()))
        set_adapter(model, True)
    items = [(i, *pr) for i, pr in enumerate(pairs)]
    opt, step = _opt(model, cfg["lr"]), 0
    model.train()
    for ep in range(cfg["epochs"]):
        for b in _batches(items, cfg["batch"], cfg["seed"], ep):
            t0 = time.perf_counter()
            k = len(b)
            tot, mean = seq_logps(model, [(p, c) for _, p, c, _ in b] + [(p, r) for _, p, _, r in b], tok.pad_token_id)
            if method == "dpo":
                rc = torch.tensor([ref[i][0] for i, *_ in b])
                rr = torch.tensor([ref[i][1] for i, *_ in b])
                loss, margin = dpo_loss(torch, tot[:k], tot[k:], rc, rr, cfg["beta"])
            else:
                loss, margin = orpo_loss(torch, mean[:k], mean[k:], cfg["lam"])
            loss = loss.mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            log.append({"stage": method, "epoch": ep, "step": step, "loss": round(loss.item(), 4),
                        "margin": round(margin.mean().item(), 4), "acc": round((margin > 0).float().mean().item(), 4),
                        "seconds": round(time.perf_counter() - t0, 3)})
    model.eval()


@torch.no_grad()
def generate(model, tok, ps: list[dict], system: str, max_new_tokens: int = 96, bs: int = 8) -> list[dict]:
    out = []
    tok.padding_side = "left"
    for i in range(0, len(ps), bs):
        chunk = ps[i: i + bs]
        enc = [encode(tok, p, system)[0] for p in chunk]
        n = max(map(len, enc))
        ids = torch.tensor([[tok.pad_token_id] * (n - len(e)) + e for e in enc])
        att = torch.tensor([[0] * (n - len(e)) + [1] * len(e) for e in enc])
        t0 = time.perf_counter()
        gen = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        dt = (time.perf_counter() - t0) / len(chunk)
        for p, g in zip(chunk, gen[:, n:]):
            toks = [t for t in g.tolist() if t != tok.pad_token_id]
            out.append({"id": p["id"], "response": tok.decode(toks, skip_special_tokens=True).strip(),
                        "tokens": len(toks), "latency_s": round(dt, 3)})
    return out


def merge(model) -> None:
    merge_lora(model)


def dump(rows: list[dict]) -> str:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
