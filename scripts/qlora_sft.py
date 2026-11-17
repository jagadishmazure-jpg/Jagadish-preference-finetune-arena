"""QLoRA SFT. ``--smoke`` trains a tiny model on CPU. The default path is the Azure ML GPU job.

The GPU path needs PEFT, TRL, bitsandbytes and a CUDA GPU. It is what infra/aml/qlora-job.yml runs.
It has not been executed in this repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prefarena import data  # noqa: E402
from prefarena.prefs import scored_candidates  # noqa: E402
from prefarena.qlora import PHI35, build_sft_trainer, smoke_flags  # noqa: E402
from prefarena.styles import DEPLOY_SYSTEM, user_message  # noqa: E402


def sft_rows(min_score: float = 0.8) -> list[dict]:
    idx = data.prompt_index()
    rows = []
    for pid, cands in sorted(scored_candidates().items()):
        if idx[pid]["split"] != "train":
            continue
        best = sorted(cands, key=lambda c: (-c["score"]["total"], len(c["response"].split())))[0]
        if best["score"]["total"] < min_score:
            continue
        prompt = idx[pid]
        text = f"{DEPLOY_SYSTEM}\n\n{user_message(prompt)}\n\nAssistant: {best['response']}"
        rows.append({"text": text})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny random model on CPU, no download")
    parser.add_argument("--model", default=PHI35["model_id"])
    parser.add_argument("--output", default="outputs/phi35-qlora")
    args = parser.parse_args()
    if args.smoke:
        flags = smoke_flags()
        print(json.dumps(flags, sort_keys=True))
        if not all(flags.values()):
            return 1
        return 0
    try:
        from datasets import Dataset
    except ImportError:
        sys.exit("GPU QLoRA needs the packages in infra/aml/environment-gpu.yml (datasets, peft, trl, bitsandbytes).")
    rows = sft_rows()
    if not rows:
        sys.exit("no SFT rows")
    trainer = build_sft_trainer(args.model, Dataset.from_list(rows), args.output)
    trainer.train()
    trainer.save_model(args.output)
    print(f"saved adapter to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
