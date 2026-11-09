"""Same test prompts, every model variant, scored by the rubric; then one judge position flip.

Runs from the committed data: python examples/compare.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefarena import data, judge  # noqa: E402
from prefarena.evaluate import VARIANTS, generations  # noqa: E402
from prefarena.prefs import best_worst_pairs  # noqa: E402
from prefarena.rubric import score  # noqa: E402


def show(text: str, n: int = 100) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def main() -> None:
    g, idx = generations(), data.prompt_index()
    test = sorted((p for p in data.prompts() if p["split"] == "test"), key=lambda p: p["id"])
    for kind in ("policy", "unsafe"):
        p = next(p for p in test if p["kind"] == kind)
        print(f"## {p['id']} ({kind}): {p['prompt']}")
        for v in VARIANTS:
            print(f"  {v:<12} {score(p, g[v][p['id']])['total']:.1f}  {show(g[v][p['id']])}")
        print()
    rows = {(r["id"], r["order"]): judge.parse(r["output"]) for r in data.rows("judge") if r["task"] == "pairs"}
    pair = next(q for q in best_worst_pairs("test") if judge.verdict(rows[(q["id"], "xy")], rows[(q["id"], "yx")]) == "tie")
    p = idx[pair["id"]]
    print(f"## Judge on {p['id']}: {p['prompt']}")
    print(f"  rubric-chosen ({score(p, pair['chosen'])['total']:.1f}): {show(pair['chosen'], 80)}")
    print(f"  rubric-rejected ({score(p, pair['rejected'])['total']:.1f}): {show(pair['rejected'], 80)}")
    print(f"  chosen shown first -> judge says {rows[(p['id'], 'xy')]}; rejected shown first -> judge says "
          f"{rows[(p['id'], 'yx')]}: the position decided, not the answer")


if __name__ == "__main__":
    main()
