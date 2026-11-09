# ADR 0001: Preferences come from a code rubric; the LLM judge is measured, not trusted

## Status

Accepted.

## Context

Preference data needs a label for every pair. Asking a model to label is cheap but brings position,
length and self-preference bias, and there are no human labels in this repo.

## Decision

`prefarena/rubric.py` scores each answer by code: the policy value, the policy ID, no invented numbers,
length, and refusal of requests that must be refused. The pair label is the higher score, with a
minimum gap. A small open model is used as a judge only to measure how far a model judge would drift
from the rubric, in both answer orders.

## Consequences

- Every label can be explained and re-computed. The rubric's blind spots are kept as failing-case
  tests (negation, codes without digits).
- The tuned model learns what the rubric rewards. That is the point here, and the reason the rubric
  is narrow and documented.
