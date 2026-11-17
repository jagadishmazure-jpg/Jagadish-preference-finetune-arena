"""PPO with the Bradley-Terry reward model.

``--smoke`` compares a tiny-model PPO run with DPO on CPU. The default path builds TRL 1.11's
``PPOTrainer`` (removed from TRL in 1.13) around ``BradleyTerryRewardModel``. That path needs a GPU
and the packages in infra/aml/environment-ppo.yml. It has not been executed in this repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prefarena.ppo import (  # noqa: E402
    BradleyTerryRewardModel,
    build_trl_ppo_trainer,
    compare_with_dpo,
    toy_groups,
)
from prefarena.qlora import PHI35  # noqa: E402


def _tiny():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model", default=PHI35["model_id"])
    parser.add_argument("--output", default="outputs/phi35-ppo")
    args = parser.parse_args()
    if args.smoke:
        _bt, groups = toy_groups()
        result = compare_with_dpo(
            groups,
            {"lora_r": 4, "lora_alpha": 8, "seed": 0, "epochs": 12, "lr": 5e-3, "batch": 2, "beta": 0.5,
             "kl_coef": 0.02, "clip": 0.2},
            _tiny,
        )
        print(json.dumps({k: result[k] for k in ("ppo_preferred_rose", "dpo_preferred_rose")}, sort_keys=True))
        if not (result["ppo_preferred_rose"] and result["dpo_preferred_rose"]):
            return 1
        return 0
    try:
        from datasets import Dataset
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("GPU PPO needs the packages in infra/aml/environment-ppo.yml.")
    from prefarena import data
    from prefarena.prefs import all_pairs
    from prefarena.reward import BT
    from prefarena.styles import DEPLOY_SYSTEM, messages

    idx = data.prompt_index()
    pairs = [(idx[p["id"]], p["win"], p["lose"]) for p in all_pairs("train")]
    bt = BT("bow").fit(pairs)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    def decode_batch(input_ids):
        items = []
        for row in input_ids.tolist():
            text = tok.decode([t for t in row if t != tok.pad_token_id], skip_special_tokens=True)
            items.append(({"prompt": text}, text))
        return items

    reward = BradleyTerryRewardModel(bt, decode_batch)
    queries = []
    for prompt in data.prompts():
        if prompt["split"] != "train":
            continue
        ids = tok.apply_chat_template(messages(prompt, DEPLOY_SYSTEM), add_generation_prompt=True, tokenize=True)
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        queries.append({"input_ids": list(ids)})
        if len(queries) >= 256:
            break
    dataset = Dataset.from_list(queries)
    trainer = build_trl_ppo_trainer(args.model, tok, dataset, reward, args.output)
    trainer.train()
    print(f"trained PPO into {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
