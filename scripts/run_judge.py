"""Run the judge (qwen2.5-1.5b, llama.cpp, CPU, greedy) in both orders. Writes data/judge.jsonl. Resumable.

Tasks:
  pairs              test-split best-vs-worst preference pairs (judge vs rubric, position and length bias)
  arena:<x>:<y>      answers of tuned variant x vs y on the test prompts (needs data/generations.jsonl)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prefarena import data, judge  # noqa: E402
from prefarena.prefs import best_worst_pairs  # noqa: E402


def tasks(which: str):
    idx = data.prompt_index()
    if which in ("pairs", "all"):
        for pr in best_worst_pairs("test"):
            yield "pairs", idx[pr["id"]], pr["chosen"], pr["rejected"]
    if which in ("arena", "all"):
        g: dict[str, dict[str, str]] = {}
        for r in data.rows("generations"):
            g.setdefault(r["variant"], {})[r["id"]] = r["response"]
        for x, y in judge.ARENA:
            for pid in sorted(g[x]):
                if idx[pid]["split"] == "test":
                    yield f"arena:{x}:{y}", idx[pid], g[x][pid], g[y][pid]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["pairs", "arena", "all"], default="all")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--threads", type=int, default=0, help="override n_threads (fewer helps on a busy machine)")
    a = ap.parse_args()
    from llama_cpp import Llama

    spec = json.loads((ROOT / "scripts/models.json").read_text())
    f = next(m["file"] for m in spec["gguf"] if m["label"] == "qwen2.5-1.5b")
    llm = Llama(str(a.models_dir / f), n_ctx=spec["sampling"]["n_ctx"], n_threads=a.threads or spec["sampling"]["n_threads"],
                seed=0, verbose=False)
    out = data.DATA / "judge.jsonl"
    done = {(r["task"], r["id"], r["order"]) for r in data.read_jsonl(out)} if out.exists() else set()
    with out.open("a", encoding="utf-8") as fh:
        for task, p, x, y in tasks(a.task):
            for order, (first, second) in (("xy", (x, y)), ("yx", (y, x))):
                if (task, p["id"], order) in done:
                    continue
                t0 = time.perf_counter()
                r = llm.create_chat_completion(messages=judge.messages(p, first, second), temperature=0,
                                               max_tokens=spec["sampling"]["judge_max_tokens"])
                fh.write(json.dumps({"task": task, "id": p["id"], "order": order,
                                     "output": r["choices"][0]["message"]["content"],
                                     "latency_s": round(time.perf_counter() - t0, 3)}) + "\n")
                fh.flush()
    print("judge done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
