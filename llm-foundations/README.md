# llm-foundations

A mini-GPT and three classic exercises. They sit next to the preference-tuning pipeline and do not
feed it: no rubric, no release rule, no measured 135M run. The preference repo's README explains
that split.

## Mini-GPT

`llmfound/minigpt.py` is a decoder-only transformer written out:

- byte-level BPE tokenizer (`llmfound/tokenizer.py`), trained on `corpus/fenwold.txt`
- token embeddings
- causal multi-head attention
- rotary positions (RoPE)
- sampling with temperature, top-k and top-p
- a KV cache for decoding
- an SVG of one attention head
- a greedy decode that checks the cache against a full forward

Training is next-token cross-entropy for a few steps on CPU. A longer run is
`azure/minigpt-job.yml` (A100 or H100). That job has not been run.

## Classic models

| Script | What it trains | Data |
|---|---|---|
| `llmfound/mlp_cnn.py` | a small MLP and a small CNN | scikit-learn digits (8 by 8), no download |
| `llmfound/intent.py` | a BiLSTM and `DistilBertForSequenceClassification` with a tiny random config | 40 synthetic shop intents |
| `llmfound/pii.py` | regex spans, and the same spans as a spaCy pipeline component | one made-up note |

`intent.py` names `distilbert-base-uncased` as the checkpoint a full run would download. The smoke
does not call `from_pretrained`. spaCy is pinned in the repo `requirements.txt`; the component does
not need a downloaded English model. It sets `doc.ents` for email, phone, payment-card number and
order id. It does not claim to find person names.

## CPU smoke

<!-- run: python scripts/foundations_smoke.py -->
```text
minigpt_loss_fell: yes
minigpt_kv_cache_matches: yes
digits_mlp_beats_majority: yes
digits_cnn_beats_majority: yes
intent_bilstm_beats_majority: yes
intent_distilbert_beats_majority: yes
pii_labels: EMAIL,ORDER_ID,PAYMENT_CARD,PHONE
pii_policy_hits: 0
```
<!-- /run -->

`minigpt_kv_cache_matches` means greedy tokens with the cache equal greedy tokens without it. The
benchmark also records both times; those times are not printed here because they change from machine
to machine. Tests check that the tokens match and that a few training steps lower the loss.

## What needs a GPU

`azure/minigpt-job.yml` trains a wider mini-GPT (width 128, 4 layers, 400 steps) on `gpu-a100`
(`Standard_NC24ads_A100_v4`). `Standard_NC40ads_H100_v5` is the alternate tag on that file. CI does
not submit it. Create the cluster only for that run, with a minimum of 0 nodes, and delete it
afterwards. List prices are in [docs/deployment.md](../docs/deployment.md).
