# ADR 0003: Measure once on CPU, commit outputs, replay in CI

## Status

Accepted.

## Context

CI must be free and deterministic. Generating candidates, judging and training take tens of minutes.

## Decision

Candidates, judge replies, embeddings, training logs and the tuned models' answers are committed and
pinned by `data/manifest.json`. CI recomputes every metric from them, and fails if any number in the
README or `results/` changes.

## Consequences

- A new run means re-running the scripts and committing new files, which shows in review.
- CI checks the method and the numbers, not that training reproduces bit for bit on other hardware.
