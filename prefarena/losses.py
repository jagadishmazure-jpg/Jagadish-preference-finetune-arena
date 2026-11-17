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


def ppo_loss(xp, logp, old_logp, advantage, clip: float = 0.2):
    """Clipped PPO surrogate. Sequence log-probs in, per-sequence advantages in.

    loss = -min(ratio * A, clip(ratio, 1-e, 1+e) * A), ratio = exp(logp - old_logp).
    old_logp is treated as a constant by the caller (detach it before the backward).
    """
    ratio = xp.exp(logp - old_logp)
    unclipped = ratio * advantage
    clipped = xp.clip(ratio, 1.0 - clip, 1.0 + clip) * advantage
    return -xp.minimum(unclipped, clipped), ratio


def kl_approx(xp, logp, ref_logp, estimator: str = "k3"):
    """Schulman KL(pi || ref) estimators used by TRL's PPO (k1 or k3). k3 is non-negative."""
    delta = logp - ref_logp
    if estimator == "k1":
        return delta
    if estimator != "k3":
        raise ValueError(f"unknown KL estimator {estimator}")
    return (xp.exp(delta) - 1) - delta
