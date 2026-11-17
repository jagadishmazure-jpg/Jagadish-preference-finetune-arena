"""CPU smoke for the mini-GPT and the classic DL/NLP scripts. Stable yes/no lines for the docs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "llm-foundations"))

from llmfound.intent import run_intent  # noqa: E402
from llmfound.minigpt import smoke  # noqa: E402
from llmfound.mlp_cnn import run_digits  # noqa: E402
from llmfound.pii import find_pii  # noqa: E402


def yn(flag: bool) -> str:
    return "yes" if flag else "no"


def main() -> int:
    gpt = smoke(steps=15)
    digits = run_digits(epochs=6)
    intent = run_intent(epochs=8)
    note = "Email mira@fenwold.example or call 415-555-0199 about ORD-104488 using card 4111 1111 1111 1111."
    labels = sorted({hit["label"] for hit in find_pii(note)})
    policy_hits = find_pii("Unused items can be returned within 45 days with a receipt.")
    lines = [
        f"minigpt_loss_fell: {yn(gpt['loss_fell'])}",
        f"minigpt_kv_cache_matches: {yn(gpt['kv_match'])}",
        f"digits_mlp_beats_majority: {yn(digits['mlp_beats_majority'])}",
        f"digits_cnn_beats_majority: {yn(digits['cnn_beats_majority'])}",
        f"intent_bilstm_beats_majority: {yn(intent['bilstm_beats_majority'])}",
        f"intent_distilbert_beats_majority: {yn(intent['distilbert_beats_majority'])}",
        f"pii_labels: {','.join(labels)}",
        f"pii_policy_hits: {len(policy_hits)}",
    ]
    print("\n".join(lines))
    ok = all(item.endswith("yes") or item.startswith("pii_") for item in lines)
    return 0 if ok and labels == ["EMAIL", "ORDER_ID", "PAYMENT_CARD", "PHONE"] and not policy_hits else 1


if __name__ == "__main__":
    sys.exit(main())
