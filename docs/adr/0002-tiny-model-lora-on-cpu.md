# ADR 0002: Tune a 135M model with hand-written LoRA on CPU

## Status

Accepted.

## Context

The goal is a real preference-tuning run anyone can repeat on a laptop, at no cost.

## Decision

SmolLM2-135M-Instruct, LoRA rank 8 on the attention projections (about 0.9M trainable parameters),
float32, responses capped at 128 tokens. LoRA is about 80 lines of PyTorch (`prefarena/lora.py`)
rather than a library, so it is easy to read and test. DPO uses the same weights with adapters off as
its reference model, computed once before training.

## Consequences

- Each run takes minutes on 8 CPU cores. The numbers describe a very small model; larger models would
  start from better answers and may gain less or more.
- Tests run the loops on a tiny random model in CI, so the code is checked without any download.
