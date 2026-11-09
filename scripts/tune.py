"""Fine-tune the tiny model on CPU and generate answers for the dev and test prompts.

Runs, all from SmolLM2-135M-Instruct with LoRA (configs below, fixed before any run):
  base          no tuning, short deploy prompt
  base_careful  no tuning, the long careful prompt (prompt-engineering baseline)
  sft           supervised tuning on the rubric-best candidate per train prompt
  dpo           SFT, merged, then DPO on train best-vs-worst pairs (reference = the SFT model)
  dpo_no_sft    DPO straight from the base model
  orpo          ORPO straight from the base model (no reference model)
Writes data/generations.jsonl, data/train_log.jsonl and data/tune_manifest.json.
Needs torch==2.15.0 and transformers==5.22.2 (requirements-train.txt). About 20-40 minutes on 8 cores.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from prefarena import data, tuning  # noqa: E402
from prefarena.prefs import best_worst_pairs, scored_candidates  # noqa: E402
from prefarena.styles import CAREFUL_SYSTEM, DEPLOY_SYSTEM  # noqa: E402

MODEL = str(ROOT / "models" / "smollm2-135m-instruct")
LORA = {"lora_r": 8, "lora_alpha": 16, "seed": 0}
CONFIGS = {
    "sft": {**LORA, "epochs": 2, "lr": 2e-4, "batch": 8, "min_score": 0.8},
    "dpo": {**LORA, "epochs": 2, "lr": 1e-4, "batch": 4, "beta": 0.1},
    "dpo_no_sft": {**LORA, "epochs": 2, "lr": 1e-4, "batch": 4, "beta": 0.1},
    "orpo": {**LORA, "epochs": 2, "lr": 2e-4, "batch": 4, "lam": 0.5},
}
MAX_NEW_TOKENS = 96


def main() -> int:
    torch.set_num_threads(8)
    idx = data.prompt_index()
    eval_ps = [p for p in data.prompts() if p["split"] in ("dev", "test")]
    tok, model = tuning.load(MODEL)
    sft_ex = []
    for pid, cands in sorted(scored_candidates().items()):
        if idx[pid]["split"] != "train":
            continue
        best = sorted(cands, key=lambda c: (-c["score"]["total"], len(c["response"].split())))[0]
        if best["score"]["total"] >= CONFIGS["sft"]["min_score"]:
            sft_ex.append(tuning.encode(tok, idx[pid], DEPLOY_SYSTEM, best["response"]))
    pairs = []
    for pr in best_worst_pairs("train"):
        p, c = tuning.encode(tok, idx[pr["id"]], DEPLOY_SYSTEM, pr["chosen"])
        pairs.append((p, c, tuning.encode(tok, idx[pr["id"]], DEPLOY_SYSTEM, pr["rejected"])[1]))
    gens, log, timing = [], [], {}

    def gen(name: str, m, system=DEPLOY_SYSTEM):
        t0 = time.perf_counter()
        for r in tuning.generate(m, tok, eval_ps, system, MAX_NEW_TOKENS):
            gens.append({"variant": name, **r})
        timing[f"generate_{name}_s"] = round(time.perf_counter() - t0, 1)
        print(name, "generated", flush=True)

    def timed(name, fn):
        t0 = time.perf_counter()
        fn()
        timing[f"train_{name}_s"] = round(time.perf_counter() - t0, 1)
        for row in log:
            row.setdefault("run", name)
        print(name, "trained", timing[f"train_{name}_s"], flush=True)

    gen("base", model)
    gen("base_careful", model, CAREFUL_SYSTEM)
    timed("sft", lambda: tuning.train_sft(model, tok, sft_ex, CONFIGS["sft"], log))
    gen("sft", model)
    tuning.merge(model)
    timed("dpo", lambda: tuning.train_pref(model, tok, pairs, CONFIGS["dpo"], log, "dpo"))
    gen("dpo", model)
    for name, method in (("dpo_no_sft", "dpo"), ("orpo", "orpo")):
        _, model = tuning.load(MODEL)
        timed(name, lambda m=model, n=name, me=method: tuning.train_pref(m, tok, pairs, CONFIGS[n], log, me))
        gen(name, model)
    (data.DATA / "generations.jsonl").write_text(tuning.dump(gens))
    (data.DATA / "train_log.jsonl").write_text(tuning.dump(log))
    manifest = {"model": "SmolLM2-135M-Instruct", "configs": CONFIGS, "sft_examples": len(sft_ex),
                "train_pairs": len(pairs), "eval_prompts": len(eval_ps), "max_new_tokens": MAX_NEW_TOKENS,
                "timing": timing, "torch": torch.__version__, "threads": torch.get_num_threads(),
                "cpu": platform.processor() or platform.machine()}
    (data.DATA / "tune_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
