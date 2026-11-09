"""Preference losses, written once for NumPy and PyTorch alike (pass the array module as xp).

All inputs are summed log-probabilities of a response given its prompt.

DPO:  -log sigmoid( beta * [(pi_c - ref_c) - (pi_r - ref_r)] )
ORPO: NLL(chosen) - lam * log sigmoid( log_odds(chosen) - log_odds(rejected) ),
      with log_odds(y) = mean_logp - log(1 - exp(mean_logp)) on the per-token mean log-prob.
"""

from __future__ import annotations


def softplus(xp, z):
    return xp.logaddexp(z * 0, z)


def dpo_loss(xp, pi_c, pi_r, ref_c, ref_r, beta: float = 0.1):
    margin = beta * ((pi_c - ref_c) - (pi_r - ref_r))
    return softplus(xp, -margin), margin


def log_odds(xp, mean_logp):
    # log(p / (1 - p)) for p = exp(mean_logp) < 1, computed stably.
    return mean_logp - xp.log(-xp.expm1(mean_logp))


def orpo_loss(xp, mean_c, mean_r, lam: float = 0.5):
    ratio = log_odds(xp, mean_c) - log_odds(xp, mean_r)
    return -mean_c + lam * softplus(xp, -ratio), ratio


def bradley_terry_prob(xp, r_a, r_b):
    """P(a preferred over b) under a Bradley-Terry reward model."""
    return 1 / (1 + xp.exp(-(r_a - r_b)))
