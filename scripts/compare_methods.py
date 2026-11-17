"""Compare hand-written LoRA, PEFT LoRA, QLoRA and Azure OpenAI fine-tuning.

CPU smoke results are measured here. Phi-3.5 VRAM and the Azure prices are arithmetic on published
figures (see prefarena/qlora.py); those jobs have not been run. Prints stable lines for the docs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prefarena import data  # noqa: E402
from prefarena.ppo import compare_with_dpo, toy_groups  # noqa: E402
from prefarena.prefs import scored_candidates  # noqa: E402
from prefarena.qlora import (  # noqa: E402
    A100_USD_PER_HOUR,
    AOAI_MODEL,
    AOAI_USD_PER_MILLION_TOKENS,
    H100_USD_PER_HOUR,
    PHI35,
    aoai_training_usd,
    lora_trainable,
    smoke_flags,
    weight_bytes,
)


def gib(n_bytes: int) -> str:
    unit = 1024 ** 3
    scaled = (n_bytes * 1000 + unit // 2) // unit
    return f"{scaled // 1000}.{scaled % 1000:03d}"


def sft_tokens() -> tuple[int, int]:
    """Rows and a character/4 token estimate for the same chosen-answer set scripts/tune.py trains on."""
    idx = data.prompt_index()
    n = chars = 0
    for pid, cands in scored_candidates().items():
        if idx[pid]["split"] != "train":
            continue
        best = sorted(cands, key=lambda c: (-c["score"]["total"], len(c["response"].split())))[0]
        if best["score"]["total"] >= 0.8:
            n += 1
            chars += len(best["response"])
    return n, (chars + 3) // 4


def tiny_model():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


def yn(flag: bool) -> str:
    return "yes" if flag else "no"


def main() -> int:
    flags = smoke_flags()
    _bt, groups = toy_groups()
    ppo = compare_with_dpo(
        groups,
        {"lora_r": 4, "lora_alpha": 8, "seed": 0, "epochs": 12, "lr": 5e-3, "batch": 2, "beta": 0.5,
         "kl_coef": 0.02, "clip": 0.2},
        tiny_model,
    )
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    rubric = metrics["variants"]["test"]["sft"]["mean_score"]
    rows, tokens = sft_tokens()
    phi_fp16 = weight_bytes(PHI35["params"], "fp16")
    phi_nf4 = weight_bytes(PHI35["params"], "nf4")
    lines = [
        f"handwritten_lora_smoke_loss_fell: {yn(flags['handwritten_loss_fell'])}",
        f"peft_lora_matches_handwritten: {yn(flags['peft_matches_handwritten'])}",
        f"qlora_smoke_loss_fell: {yn(flags['qlora_loss_fell'])}",
        f"ppo_preferred_rose: {yn(ppo['ppo_preferred_rose'])}",
        f"dpo_preferred_rose: {yn(ppo['dpo_preferred_rose'])}",
        f"smollm_sft_test_mean_rubric: {rubric:.4f}",
        "smollm_lora_trainable: 921600",
        f"phi35_params: {PHI35['params']}",
        f"phi35_fp16_gib: {gib(phi_fp16)}",
        f"phi35_nf4_gib: {gib(phi_nf4)}",
        f"phi35_lora_trainable: {lora_trainable(PHI35['layers'], PHI35['hidden'], 16)}",
        f"a100_usd_per_hour: {A100_USD_PER_HOUR:.3f}",
        f"h100_usd_per_hour: {H100_USD_PER_HOUR:.3f}",
        f"aoai_model: {AOAI_MODEL}",
        f"aoai_usd_per_million_tokens: {AOAI_USD_PER_MILLION_TOKENS:.2f}",
        f"aoai_sft_rows: {rows}",
        f"aoai_sft_tokens_est: {tokens}",
        f"aoai_3_epoch_usd: {aoai_training_usd(tokens, 3):.2f}",
        "gpu_phi35_accuracy: not_measured",
        "aoai_accuracy: not_measured",
    ]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
