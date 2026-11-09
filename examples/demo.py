"""Walk through one test prompt: four candidates, rubric scores, the preference pair, and hacking probes.

Runs from the committed data in under a second: python examples/demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefarena import data  # noqa: E402
from prefarena.hacks import probes  # noqa: E402
from prefarena.prefs import best_worst_pairs, scored_candidates  # noqa: E402
from prefarena.rubric import score  # noqa: E402


def show(text: str, n: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def main() -> None:
    idx = data.prompt_index()
    pair = best_worst_pairs("test")[0]
    p = idx[pair["id"]]
    print(f"Prompt {p['id']} ({p['kind']}, policy {p['policy']}): {p['prompt']}\n")
    for c in scored_candidates()[p["id"]]:
        print(f"  {c['style']:<10} {c['model']:<13} score {c['score']['total']:.1f}  {show(c['response'])}")
    print(f"\nPair: chosen={pair['chosen_style']} rejected={pair['rejected_style']} gap={pair['gap']}")
    print("\nProbes built from the chosen answer (rubric score must drop):")
    for kind, text in probes(p, pair["chosen"]).items():
        print(f"  {kind:<13} {score(p, text)['total']:.1f}  {show(text, 80)}")


if __name__ == "__main__":
    main()
