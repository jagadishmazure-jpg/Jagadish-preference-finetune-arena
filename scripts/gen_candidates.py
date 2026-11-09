"""Generate four candidate responses per prompt with small open models on CPU (llama.cpp).

Writes data/candidates.jsonl (resumable). Needs llama-cpp-python==0.3.36 and the GGUF files from
scripts/fetch_models.sh. One model is loaded at a time to keep memory low.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prefarena import data  # noqa: E402
from prefarena.company import rng  # noqa: E402
from prefarena.styles import STYLES, messages  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="override n_threads (fewer helps on a busy machine)")
    a = ap.parse_args()
    from llama_cpp import Llama

    spec = json.loads((ROOT / "scripts/models.json").read_text())
    files = {m["label"]: m["file"] for m in spec["gguf"]}
    s = spec["sampling"]
    out = data.DATA / "candidates.jsonl"
    done = {(r["id"], r["style"]) for r in data.read_jsonl(out)} if out.exists() else set()
    ps = data.prompts()[: a.limit or None]
    for label in sorted({v[0] for v in STYLES.values()}):
        llm = Llama(str(a.models_dir / files[label]), n_ctx=s["n_ctx"], n_threads=a.threads or s["n_threads"], seed=0, verbose=False)
        with out.open("a", encoding="utf-8") as fh:
            for p in ps:
                for style, (model, temp, system, ctx) in STYLES.items():
                    if model != label or (p["id"], style) in done:
                        continue
                    seed = rng("cand", p["id"], style).randrange(2**31)
                    t0 = time.perf_counter()
                    r = llm.create_chat_completion(messages=messages(p, system, ctx), temperature=temp, seed=seed,
                                                   max_tokens=s["candidate_max_tokens"])
                    fh.write(json.dumps({"id": p["id"], "style": style, "model": model, "seed": seed,
                                         "response": r["choices"][0]["message"]["content"].strip(),
                                         "completion_tokens": r["usage"]["completion_tokens"],
                                         "latency_s": round(time.perf_counter() - t0, 3)}) + "\n")
                    fh.flush()
        del llm
        print(label, "done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
