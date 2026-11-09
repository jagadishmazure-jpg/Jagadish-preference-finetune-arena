"""Preference losses (NumPy here, the same code runs on PyTorch) and the statistics helpers."""

import math

import numpy as np
import pytest

from prefarena.losses import bradley_terry_prob, dpo_loss, log_odds, orpo_loss
from prefarena.metrics import cohen_kappa, mcnemar_exact, percentile, rate, wilson


def test_dpo_loss_is_log2_when_policy_equals_reference():
    loss, margin = dpo_loss(np, np.array([-5.0]), np.array([-7.0]), np.array([-5.0]), np.array([-7.0]))
    assert margin[0] == 0 and loss[0] == pytest.approx(math.log(2))


def test_dpo_loss_falls_as_the_chosen_answer_gains_probability():
    ref_c, ref_r = np.array([-10.0]), np.array([-10.0])
    losses = [dpo_loss(np, np.array([c]), np.array([-10.0]), ref_c, ref_r, beta=0.1)[0][0] for c in (-12, -10, -8, -4)]
    assert losses == sorted(losses, reverse=True)


def test_dpo_matches_the_formula():
    loss, m = dpo_loss(np, np.array([-3.0]), np.array([-9.0]), np.array([-4.0]), np.array([-6.0]), beta=0.5)
    z = 0.5 * ((-3 + 4) - (-9 + 6))
    assert m[0] == pytest.approx(z) and loss[0] == pytest.approx(-math.log(1 / (1 + math.exp(-z))))


def test_orpo_log_odds_is_stable_near_zero_and_loss_prefers_chosen():
    assert np.isfinite(log_odds(np, np.array([-1e-9, -50.0]))).all()
    good, _ = orpo_loss(np, np.array([-0.5]), np.array([-3.0]))
    bad, _ = orpo_loss(np, np.array([-3.0]), np.array([-0.5]))
    assert good[0] < bad[0]


def test_bradley_terry_is_symmetric():
    assert bradley_terry_prob(np, 2.0, 1.0) + bradley_terry_prob(np, 1.0, 2.0) == pytest.approx(1.0)


def test_wilson_rate_and_sign_test():
    lo, hi = wilson(0, 20)
    assert lo == pytest.approx(0) and 0.15 < hi < 0.17
    assert rate(30, 40)["rate"] == 0.75
    assert mcnemar_exact(0, 8) == pytest.approx(0.0078125) and mcnemar_exact(5, 5) == 1.0


def test_cohen_kappa_and_percentile():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
    assert percentile([1, 2, 3, 4], 50) == 2.5
    with pytest.raises(ValueError):
        cohen_kappa([], [])
