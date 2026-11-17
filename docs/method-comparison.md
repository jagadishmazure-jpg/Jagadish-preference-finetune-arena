# Hand-written LoRA, PEFT LoRA, QLoRA, PPO and managed fine-tuning

The 135M preference run is unchanged. This page is the extra comparison: what CI actually measures on
CPU, and what is only arithmetic until someone pays for a GPU.

The committed SmolLM2 SFT model scores
0.8235<!-- m:variants/test/sft/mean_score --> on the test rubric. That is the only accuracy number
here that came from a real training run. Phi-3.5 QLoRA, PEFT on that model, and Azure OpenAI
fine-tuning have not been run, so their accuracy is `not_measured`.

## What the CPU smoke prints

<!-- run: python scripts/compare_methods.py -->
```text
handwritten_lora_smoke_loss_fell: yes
peft_lora_matches_handwritten: yes
qlora_smoke_loss_fell: yes
ppo_preferred_rose: yes
dpo_preferred_rose: yes
smollm_sft_test_mean_rubric: 0.8235
smollm_lora_trainable: 921600
phi35_params: 3821079552
phi35_fp16_gib: 7.117
phi35_nf4_gib: 1.836
phi35_lora_trainable: 12582912
a100_usd_per_hour: 3.673
h100_usd_per_hour: 6.980
aoai_model: gpt-4o-mini
aoai_usd_per_million_tokens: 3.30
aoai_sft_rows: 203
aoai_sft_tokens_est: 5418
aoai_3_epoch_usd: 0.05
gpu_phi35_accuracy: not_measured
aoai_accuracy: not_measured
```
<!-- /run -->

## How to read the lines

| Line | Meaning |
|---|---|
| `handwritten_lora_smoke_loss_fell` | A few SFT steps on a tiny random LM with `prefarena/lora.py` reduce the loss. |
| `peft_lora_matches_handwritten` | The PEFT forward (`y = Wx + x A B * alpha/r`, B starts at 0, dropout 0) matches the hand-written adapter on the same seed, including the final loss. |
| `qlora_smoke_loss_fell` | The base linears are replaced by an NF4 reconstruction (4-bit codes, double-quantised scales) and LoRA still reduces the loss. The matmul is fp32. bitsandbytes does the real 4-bit matmul on the GPU job. |
| `ppo_preferred_rose` / `dpo_preferred_rose` | On a fixed set of answers, both PPO and DPO raise the probability of the completion the Bradley-Terry model scores higher. PPO samples; DPO uses the pair. |
| `phi35_fp16_gib` / `phi35_nf4_gib` | Published parameter count times 2 bytes, or this repo's NF4 storage formula (4-bit codes plus blockwise 8-bit scales). bitsandbytes stores those scales in FP8, so a GPU footprint is close to this figure and not identical. This is weight storage, not measured VRAM. Activations and Adam states are extra. Rank-16 attention LoRA is about 12.6M trainable parameters. |
| `a100_usd_per_hour` / `h100_usd_per_hour` | Linux pay-as-you-go list price, eastus, checked 2026-11-17. Not a bill. |
| `aoai_3_epoch_usd` | 203 chosen answers. The token count is `characters // 4` of the answer text only, not the policy prompt, so a real JSONL file costs more. 3 epochs at the published GPT-4o mini training rate of $3.30 per million tokens. Recheck Foundry before submitting. |
| `gpu_phi35_accuracy` / `aoai_accuracy` | Not measured. A GPU job is not a substitute for the rubric, the safety rule, or the held-out split. |

## What still needs a machine

| Path | Needs | Status |
|---|---|---|
| `python scripts/qlora_sft.py --smoke` and `python scripts/ppo_trl.py --smoke` | CPU, the repo requirements | Run in CI |
| `infra/aml/qlora-job.yml` | A100 or H100, PEFT, TRL 1.18.1, bitsandbytes | Written, not run |
| `infra/aml/qlora-model.yml`, `qlora-endpoint.yml`, `qlora-deployment.yml` | The adapter from that job. The endpoint does not scale to zero | Written, not run |
| `infra/aml/ppo-job.yml` | A100 or H100, TRL 1.11.0 (`PPOTrainer` was removed in TRL 1.13) | Written, not run |
| `infra/aml/aoai-finetune.yml` | A Foundry fine-tune of `gpt-4o-mini` | Written, not run |

PPO's reward on the GPU path is `reward.BT` fitted on the training pairs, exposed as the module
TRL's `get_reward` calls (`base_model_prefix` and `score`). The smoke uses the same class on a
handful of answers so CI does not need TRL.
