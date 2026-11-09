"""Turn scored candidates into preference pairs.

For each prompt, the four candidates are scored by the rubric. The training pair is the best against
the worst, kept only when the score gap is at least PAIR_GAP. Ties among the best go to the shorter
response, then to style order, so the choice never depends on file order. All ordered pairs with a gap
of at least rubric.PREFER_GAP are also kept, for the reward models.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from . import data
from .rubric import PREFER_GAP, score, words
from .styles import STYLES

PAIR_GAP = 0.2
STYLE_ORDER = list(STYLES)


def scored_candidates() -> dict[str, list[dict]]:
    idx = data.prompt_index()
    out: dict[str, list[dict]] = defaultdict(list)
    for c in data.rows("candidates"):
        out[c["id"]].append({**c, "score": score(idx[c["id"]], c["response"])})
    for v in out.values():
        v.sort(key=lambda c: STYLE_ORDER.index(c["style"]))
    return dict(out)


def _rank_key(c: dict):
    return (-c["score"]["total"], words(c["response"]), STYLE_ORDER.index(c["style"]))


def best_worst_pairs(split: str | None = None) -> list[dict]:
    idx = data.prompt_index()
    pairs = []
    for pid, cands in sorted(scored_candidates().items()):
        if split and idx[pid]["split"] != split:
            continue
        ranked = sorted(cands, key=_rank_key)
        best, worst = ranked[0], ranked[-1]
        if round(best["score"]["total"] - worst["score"]["total"], 4) >= PAIR_GAP:
            pairs.append({"id": pid, "chosen": best["response"], "rejected": worst["response"],
                          "chosen_style": best["style"], "rejected_style": worst["style"],
                          "gap": round(best["score"]["total"] - worst["score"]["total"], 4)})
    return pairs


def all_pairs(split: str | None = None) -> list[dict]:
    """Every candidate pair with a clear rubric preference, winner first."""
    idx = data.prompt_index()
    out = []
    for pid, cands in sorted(scored_candidates().items()):
        if split and idx[pid]["split"] != split:
            continue
        for a, b in combinations(cands, 2):
            d = round(a["score"]["total"] - b["score"]["total"], 4)
            if abs(d) >= PREFER_GAP:
                w, l_ = (a, b) if d > 0 else (b, a)
                out.append({"id": pid, "win": w["response"], "lose": l_["response"],
                            "win_style": w["style"], "lose_style": l_["style"]})
    return out
