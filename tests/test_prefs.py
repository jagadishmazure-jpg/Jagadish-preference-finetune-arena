"""Preference pairs built from the committed candidates follow their stated rules."""

from prefarena import data
from prefarena.prefs import PAIR_GAP, all_pairs, best_worst_pairs, scored_candidates
from prefarena.rubric import score
from prefarena.styles import STYLES


def test_every_prompt_has_one_candidate_per_style():
    sc = scored_candidates()
    assert len(sc) == len(data.prompts())
    assert all([c["style"] for c in cs] == list(STYLES) for cs in sc.values())


def test_training_pairs_respect_the_gap_and_the_rubric():
    idx = data.prompt_index()
    for p in best_worst_pairs("train"):
        sc_c, sc_r = score(idx[p["id"]], p["chosen"])["total"], score(idx[p["id"]], p["rejected"])["total"]
        assert round(sc_c - sc_r, 4) == p["gap"] >= PAIR_GAP


def test_ties_at_the_top_go_to_the_shorter_answer():
    from prefarena.rubric import words

    sc = scored_candidates()
    for p in best_worst_pairs("train"):
        top = max(c["score"]["total"] for c in sc[p["id"]])
        tied = [c for c in sc[p["id"]] if c["score"]["total"] == top]
        assert words(p["chosen"]) == min(words(c["response"]) for c in tied)


def test_all_pairs_put_the_higher_score_first():
    idx = data.prompt_index()
    for p in all_pairs("test")[:200]:
        assert score(idx[p["id"]], p["win"])["total"] > score(idx[p["id"]], p["lose"])["total"]


def test_pairs_never_leak_across_splits():
    idx = data.prompt_index()
    for split in ("train", "dev", "test"):
        assert {idx[p["id"]]["split"] for p in best_worst_pairs(split)} == {split}
