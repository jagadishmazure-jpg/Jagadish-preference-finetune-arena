"""Bradley-Terry reward models and the committed embeddings they use."""

import numpy as np
from conftest import policy_prompt

from prefarena import data
from prefarena.reward import BT, embed_text, embeddings, key, length_rule

P = policy_prompt()
GOOD = ["Returns are accepted within 45 days [RET-1].", "You have 45 days to return it [RET-1].",
        "Within 45 days with a receipt [RET-1]."]
BAD = ["I am not sure, maybe a while.", "Probably 90 days or so.", "Sorry, I can't help."]


def toy():
    return [(P, g, b) for g in GOOD for b in BAD]


def test_bow_model_learns_toy_preferences():
    rm = BT("bow").fit(toy())
    r = rm.reward([(P, t) for t in GOOD + BAD])
    assert r[:3].min() > r[3:].max()


def test_reward_is_a_difference_model_with_no_offset():
    rm = BT("bow").fit(toy())
    assert rm.lr.intercept_ == 0  # no intercept: only differences between answers matter
    a, b = rm.reward([(P, GOOD[0]), (P, BAD[0])])
    assert 1 / (1 + np.exp(-(a - b))) > 0.5


def test_length_rules():
    items = [(P, "one two"), (P, "one two three")]
    assert list(length_rule(True)(items)) == [2, 3] and list(length_rule(False)(items)) == [-2, -3]


def test_embedding_keys_are_stable_hashes():
    assert key("abc") == key("abc") != key("abd") and len(key("abc")) == 20
    assert embed_text(P, "x").endswith("Assistant: x") and "[RET-1]" in embed_text(P, "x")


def test_committed_embeddings_cover_every_candidate_and_generation():
    e, idx = embeddings(), data.prompt_index()
    need = {key(embed_text(idx[c["id"]], c["response"])) for c in data.rows("candidates")}
    need |= {key(embed_text(idx[g["id"]], g["response"])) for g in data.rows("generations")}
    assert need <= set(e)
    v = next(iter(e.values()))
    assert v.shape == (384,) and abs(float(np.linalg.norm(v)) - 1) < 0.01
