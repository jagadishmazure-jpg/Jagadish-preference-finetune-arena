# ADR 0007: QLoRA and PPO stay next to preference tuning; the mini-GPT does not

## Status

Accepted.

## Context

The portfolio needed a PEFT/TRL QLoRA path, PPO next to DPO, a transformer written out, and a few
classic deep-learning examples. The measured preference pipeline is a 135M model on CPU. Phi-3.5-mini
in 4-bit does not fit that run, and a from-scratch transformer is a different lesson.

## Decision

QLoRA and PPO live in `prefarena/` because they use the same prompts, the same reward model and the
same LoRA targets as the preference run. The GPU command jobs, the adapter registration and the
online endpoint are YAML under `infra/aml/`. They are not applied. CI runs a tiny-model smoke:
NF4 round-trip, PEFT formula against the hand-written adapter, and PPO against DPO.

TRL 1.13 removed `PPOTrainer`. The PPO job pins TRL 1.11.0, where it still lives at
`trl.experimental.ppo`. The QLoRA job uses current TRL (`SFTTrainer`).

The mini-GPT, the digits MLP/CNN, the BiLSTM versus a tiny DistilBERT, and the spaCy PII component
live in `llm-foundations/` with their own README. They do not import the rubric and they do not
change the release rule.

## Consequences

- Accuracy, cost and VRAM for Phi-3.5 and for managed fine-tuning are either formula estimates or
  marked not measured. The smoke only shows that the losses move on a tiny model.
- An online endpoint bills at least one GPU and does not scale to zero. The YAML says so.
- Someone who only wants the preference story can ignore `llm-foundations/`.
