"""Sequence-level PPO against the Bradley-Terry reward model, and a TRL PPOTrainer builder.

TRL removed ``PPOTrainer`` in 1.13.0 (it had already left the top-level ``trl`` package in 1.10).
The GPU job pins ``trl==1.11.0`` and imports ``trl.experimental.ppo``, the last release that still
ships it. CI does not install TRL. The smoke below uses the same clipped surrogate
(``losses.ppo_loss``) and the same reward model (``reward.BT``) on a tiny random LM, so the loop
is tested without a download.

The smoke is a contextual bandit: each prompt has a fixed set of completions, the policy is the
tiny LM's sequence log-probability, and the advantage is the Bradley-Terry reward minus the batch
mean. TRL's trainer also fits a value head and samples tokens; that path is ``build_trl_ppo_trainer``.
"""

from __future__ import annotations

import torch
from torch import nn

from .lora import apply_lora, set_adapter
from .losses import kl_approx, ppo_loss
from .reward import BT
from .tuning import seq_logps

PPO_CLIP = 0.2
PPO_KL_COEF = 0.05


def load_trl_ppo():
    """``(PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead)`` from the TRL that still has PPO."""
    try:
        from trl.experimental.ppo import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
    except ImportError:
        from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer  # TRL <= 1.9
    return PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead


def ppo_config_kwargs(output_dir: str, batch: int = 2, response_length: int = 64) -> dict:
    """Arguments for ``trl.experimental.ppo.PPOConfig`` at TRL 1.11."""
    return {
        "output_dir": output_dir,
        "learning_rate": 3e-6,
        "per_device_train_batch_size": batch,
        "gradient_accumulation_steps": 1,
        "num_ppo_epochs": 1,
        "num_mini_batches": 1,
        "total_episodes": 1000,
        "response_length": response_length,
        "temperature": 0.7,
        "kl_coef": PPO_KL_COEF,
        "cliprange": PPO_CLIP,
        "whiten_rewards": False,
        "missing_eos_penalty": 1.0,
        "report_to": "none",
        "seed": 0,
        "bf16": True,
        "logging_steps": 1,
    }


class _IdBackbone(nn.Module):
    """Stand-in for the transformer body TRL's ``get_reward`` runs before ``model.score``.

    The token ids travel inside the hidden state this forward returns. ``score`` reads them from
    that tensor, so two overlapping calls cannot swap each other's batches.
    """

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, output_hidden_states=False,
                return_dict=True, use_cache=False, **_kwargs):
        hidden = input_ids.float().unsqueeze(-1)

        class _Out:
            pass

        out = _Out()
        out.hidden_states = (hidden,)
        return out


def completion_text(decoded: str, marker: str = "<|assistant|>") -> str:
    """The generated answer after the chat template's assistant marker.

    The Bradley-Terry model was fitted on answer text. Scoring the whole query plus the answer
    would let the prompt's words move the reward.
    """
    tail = decoded.rsplit(marker, 1)[1] if marker in decoded else decoded
    for stop in ("<|end|>", "<|endoftext|>"):
        if stop in tail:
            tail = tail.split(stop, 1)[0]
    return tail.strip()


class BradleyTerryRewardModel(nn.Module):
    """TRL-shaped reward module (``base_model_prefix`` + ``score``) backed by ``reward.BT``.

    TRL 1.11's ``get_reward`` does ``getattr(model, model.base_model_prefix)(...)`` and then
    ``model.score(hidden_states)``, and reads the logit at the last non-pad token. This module
    decodes those token ids with ``decode_batch`` and returns the Bradley-Terry reward at every
    position, so that index is the reward of the decoded completion.
    """

    base_model_prefix = "backbone"

    def __init__(self, bt: BT, decode_batch):
        super().__init__()
        self.bt = bt
        self.decode_batch = decode_batch
        self.backbone = _IdBackbone()

    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ids = hidden_states.detach().squeeze(-1).round().to(dtype=torch.long)
        items = self.decode_batch(ids)
        values = torch.as_tensor(self.bt.reward(items), dtype=hidden_states.dtype, device=hidden_states.device)
        return values.reshape(-1, 1, 1).expand(-1, hidden_states.shape[1], 1).contiguous()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs)
        return self.score(out.hidden_states[-1])


def reward_at_last_token(model: BradleyTerryRewardModel, input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """The score TRL keeps: logit at the last non-pad position of each row."""
    logits = model(input_ids=input_ids, attention_mask=input_ids.ne(pad_id))
    last = []
    for row, logit in zip(input_ids, logits):
        positions = (row != pad_id).nonzero()
        index = int(positions[-1]) if len(positions) else 0
        last.append(logit[index, 0])
    return torch.stack(last)


def build_trl_ppo_trainer(model_id: str, tokenizer, train_dataset, reward_model, output_dir: str):
    """Construct TRL's PPO trainer. Imports TRL; the CPU smoke never calls this."""
    PPOConfig, PPOTrainer, _ValueHead = load_trl_ppo()
    from .qlora import build_peft_config

    policy = model_id
    # A value head on the same base. TRL 1.11 requires this module; it is not the Bradley-Terry model.
    value_model = _ValueHead.from_pretrained(model_id)
    config = PPOConfig(**ppo_config_kwargs(output_dir))
    return PPOTrainer(
        args=config,
        processing_class=tokenizer,
        model=policy,
        ref_model=None,
        reward_model=reward_model,
        train_dataset=train_dataset,
        value_model=value_model,
        peft_config=build_peft_config(model_id),
    )


def _logps(model, group, pad: int) -> torch.Tensor:
    return seq_logps(model, [(c["prompt_ids"], c["response_ids"]) for c in group], pad)[0]


def _pref_prob(model, groups, pad: int) -> float:
    """Mean probability the policy assigns to the completion with the higher reward, within its group."""
    total = 0.0
    with torch.no_grad():
        for group in groups:
            lp = _logps(model, group, pad)
            best = max(range(len(group)), key=lambda i: group[i]["reward"])
            total += float(torch.softmax(lp, dim=0)[best])
    return total / len(groups)


def _greedy_reward(model, groups, pad: int) -> float:
    total = 0.0
    with torch.no_grad():
        for group in groups:
            lp = _logps(model, group, pad)
            total += group[int(lp.argmax())]["reward"]
    return total / len(groups)


def train_ppo(model, groups: list[list[dict]], cfg: dict, log: list, pad: int = 0) -> None:
    """PPO on a fixed completion set. Each candidate dict needs prompt_ids, response_ids and reward.

    One rollout samples a completion per prompt and freezes its log-probability. Several optimiser
    steps then reuse that rollout, so the probability ratio can leave 1 and the clip can engage.
    The next rollout is collected only after those steps.
    """
    torch.manual_seed(cfg["seed"])
    apply_lora(model, r=cfg["lora_r"], alpha=cfg["lora_alpha"])
    set_adapter(model, False)
    with torch.no_grad():
        ref = [_logps(model, group, pad).tolist() for group in groups]
    set_adapter(model, True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=0.0)
    generator = torch.Generator()
    generator.manual_seed(cfg["seed"])
    model.train()
    step = 0
    clip = cfg.get("clip", PPO_CLIP)
    for _epoch in range(cfg["epochs"]):
        with torch.no_grad():
            detached = [_logps(model, group, pad) for group in groups]
        choices = []
        for lp in detached:
            probs = torch.softmax(lp, dim=0)
            choices.append(int(torch.multinomial(probs, 1, generator=generator)))
        batch = [(groups[i][j]["prompt_ids"], groups[i][j]["response_ids"]) for i, j in enumerate(choices)]
        old_lp = torch.tensor([float(detached[i][j]) for i, j in enumerate(choices)])
        ref_lp = torch.tensor([ref[i][j] for i, j in enumerate(choices)])
        rewards = torch.tensor([groups[i][j]["reward"] for i, j in enumerate(choices)])
        advantage = rewards - rewards.mean()
        for _inner in range(cfg.get("ppo_epochs", 4)):
            new_lp, _mean = seq_logps(model, batch, pad)
            surrogate, ratio = ppo_loss(torch, new_lp, old_lp.to(new_lp.dtype), advantage.to(new_lp.dtype), clip)
            kl = kl_approx(torch, new_lp, ref_lp.to(new_lp.dtype), "k3")
            loss = surrogate.mean() + cfg.get("kl_coef", PPO_KL_COEF) * kl.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            log.append({
                "stage": "ppo",
                "step": step,
                "loss": round(float(loss.detach()), 4),
                "reward": round(float(rewards.mean()), 4),
                "kl": round(float(kl.mean().detach()), 4),
                "ratio": round(float(ratio.detach().mean()), 4),
                "clipfrac": round(float((ratio.detach() - 1).abs().gt(clip - 1e-6).float().mean()), 4),
            })
    model.eval()


def compare_with_dpo(groups: list[list[dict]], cfg: dict, make_model, pad: int = 0) -> dict:
    """Train PPO and DPO from the same tiny initialisation and report whether each raised the
    log-prob of the completion the reward model prefers."""
    from .tuning import train_pref

    ppo_model, dpo_model = make_model(), make_model()
    ppo_start = _pref_prob(ppo_model, groups, pad)
    dpo_start = _pref_prob(dpo_model, groups, pad)
    ppo_log: list[dict] = []
    train_ppo(ppo_model, groups, cfg, ppo_log, pad)
    pairs = []
    for group in groups:
        ranked = sorted(group, key=lambda c: c["reward"], reverse=True)
        pairs.append((ranked[0]["prompt_ids"], ranked[0]["response_ids"], ranked[1]["response_ids"]))
    dpo_log: list[dict] = []
    train_pref(dpo_model, type("T", (), {"pad_token_id": pad})(), pairs,
               {**cfg, "beta": cfg.get("beta", 0.5)}, dpo_log, "dpo")
    return {
        "ppo_preferred_rose": _pref_prob(ppo_model, groups, pad) > ppo_start,
        "dpo_preferred_rose": _pref_prob(dpo_model, groups, pad) > dpo_start,
        "ppo_greedy_reward": round(_greedy_reward(ppo_model, groups, pad), 4),
        "dpo_greedy_reward": round(_greedy_reward(dpo_model, groups, pad), 4),
        "dpo_final_pair_acc": dpo_log[-1]["acc"],
        "ppo_steps": len(ppo_log),
        "ppo_final_loss": ppo_log[-1]["loss"],
        "dpo_final_loss": dpo_log[-1]["loss"],
    }


def toy_groups() -> tuple[BT, list[list[dict]]]:
    """A Bradley-Terry model on a handful of shop answers, and the token sequences PPO/DPO train on."""
    prompt = {
        "id": "t", "kind": "policy", "policy": "RET-1", "topic": "returns",
        "prompt": "How long do I have to return a fleece?", "split": "train",
    }
    good = [
        "Returns are accepted within 45 days [RET-1].",
        "You have 45 days to return it [RET-1].",
        "Within 45 days with a receipt [RET-1].",
    ]
    bad = [
        "I am not sure, maybe a while.",
        "Probably 90 days or so.",
        "Sorry, I can't help.",
    ]
    bt = BT("bow").fit([(prompt, g, b) for g in good for b in bad])
    groups = []
    for i, (g, b) in enumerate(zip(good, bad)):
        prefix = [1, 2, 3 + i]
        chosen = {"prompt": prompt, "text": g, "prompt_ids": prefix, "response_ids": [10, 11, 12, 13],
                  "reward": float(bt.reward([(prompt, g)])[0])}
        rejected = {"prompt": prompt, "text": b, "prompt_ids": prefix, "response_ids": [20, 21, 22, 23],
                    "reward": float(bt.reward([(prompt, b)])[0])}
        groups.append([chosen, rejected])
    return bt, groups
