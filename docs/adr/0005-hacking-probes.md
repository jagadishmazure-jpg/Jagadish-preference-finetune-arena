# ADR 0005: Check reward hacking with probes the rubric already rejects

## Status

Accepted.

## Context

A reward model can learn shortcuts (length, keywords). A policy tuned against it can exploit them.

## Decision

`prefarena/hacks.py` edits good answers in ways the rubric scores lower: padding with polite filler,
stuffing the policy ID, changing the number, refusing a fair question, or complying with an unsafe
request. Each reward model is checked on how often it prefers the edit. The tuned models' answers are
also checked for the same signs (length growth, ID stuffing, over-refusal).

## Consequences

- Shortcuts are measured, not assumed. The probes cover known shortcuts only; new ones need new probes.
