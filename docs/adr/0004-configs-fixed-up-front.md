# ADR 0004: Training settings fixed before the first run; no tuning on test

## Status

Accepted.

## Context

With several methods it is easy to keep changing settings until test numbers look good.

## Decision

All settings in `scripts/tune.py` (epochs, learning rates, beta, lambda, LoRA rank) were written
before the first training run and not changed after seeing results. Dev numbers are reported next to
test numbers so a reader can see whether they agree.

## Consequences

- Some methods may be under-tuned. That is reported as a limitation rather than fixed by looking at test.
