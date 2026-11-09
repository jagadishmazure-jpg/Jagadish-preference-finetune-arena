# Implementation guide

How to rebuild this project from scratch, in the order it was built. Every step runs on a laptop CPU.

## 1. Prompts and policies (`prefarena/company.py`, `prefarena/prompts.py`)

1. Write 13 short store policies, each with an ID (`RET-1`) and the exact values an answer must state.
2. Write 4 or 5 question templates per policy, and 3 templates for each of four unsafe kinds
   (other customers' data, rule exceptions, staff secrets, out-of-scope help).
3. Expand each template with polite prefixes and suffixes, remove duplicates, and split by template
   hash so no wording appears in two splits. `scripts/build_dataset.py --check` proves the file matches.

## 2. Candidates (`scripts/gen_candidates.py`)

Four answers per prompt from two small open models (llama.cpp, CPU) under four system prompts:
careful (with the policy, temperature 0), plain, chatty (long and eager) and no_context (no policy).
The point is a spread of quality, not good answers. The script is resumable.

## 3. Labels (`prefarena/rubric.py`, `prefarena/prefs.py`)

Score every answer by code. The training pair is the best against the worst answer when the gap is at
least 0.2; ties at the top go to the shorter answer. All clear pairs (gap 0.1 or more) train the reward
models.

## 4. Reward models (`prefarena/reward.py`, `scripts/embed.py`)

Bradley-Terry as logistic regression on feature differences in both orders with no intercept:
TF-IDF words, and small sentence embeddings computed once and committed. Compare with the two length
rules, then run the hacking probes (`prefarena/hacks.py`).

## 5. Tuning (`prefarena/lora.py`, `prefarena/tuning.py`, `scripts/tune.py`)

1. `apply_lora` swaps attention projections for `LoRALinear` (B starts at zero, so the model is
   unchanged at step 0; tested).
2. SFT on chosen answers that score 0.8 or more; merge; DPO from the merged model, with the reference
   log-probabilities computed once with adapters off.
3. Two ablations: DPO straight from base, and ORPO (SFT and odds-ratio terms in one loss) from base.
4. Greedy answers on dev and test prompts for every variant, including base with the careful prompt.

## 6. Judge (`prefarena/judge.py`, `scripts/run_judge.py`)

Ask a small open model which answer is better, in both orders. Count position flips, how often it
picks the first answer, agreement and Cohen's kappa against the rubric, length pull, and agreement by
which model wrote the rubric's choice.

## 7. Evaluation and gate (`prefarena/evaluate.py`, `scripts/run_eval.py`)

Rubric win rates with Wilson intervals and sign tests, the safety regression set, prompt copying and
ID stuffing, reward-model preferences on tuned answers, training curves, and the release rule on dev
(ADR 0006). The gate fails CI if the release candidate regresses.

## 8. Replay

`scripts/manifest.py` writes SHA-256 of every measured file. CI checks the manifest, recomputes every
metric, and checks that each number in the docs matches.
