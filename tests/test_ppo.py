"""PPO on CPU: the clipped loss, the Bradley-Terry reward module, and a comparison with DPO."""

import pytest

torch = pytest.importorskip("torch")

from conftest import policy_prompt  # noqa: E402

from prefarena.losses import ppo_loss  # noqa: E402
from prefarena.ppo import BradleyTerryRewardModel, compare_with_dpo, reward_at_last_token, toy_groups, train_ppo  # noqa: E402
from prefarena.reward import BT  # noqa: E402


def _tiny():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


def test_torch_ppo_loss_matches_the_numpy_formula():
    import numpy as np

    args = (np.array([-1.0, -2.0]), np.array([-1.2, -1.5]), np.array([0.5, -0.25]))
    n_loss, n_ratio = ppo_loss(np, *args, clip=0.2)
    t_loss, t_ratio = ppo_loss(torch, *(torch.tensor(a) for a in args), clip=0.2)
    assert t_loss.tolist() == pytest.approx(n_loss.tolist())
    assert t_ratio.tolist() == pytest.approx(n_ratio.tolist())


def test_reward_module_returns_the_bradley_terry_score_at_the_last_token():
    prompt = policy_prompt()
    good, bad = "Returns are accepted within 45 days [RET-1].", "I am not sure, maybe a while."
    bt = BT("bow").fit([(prompt, good, bad)] * 4)
    table = {1: (prompt, good), 2: (prompt, bad)}

    def decode(ids):
        # One completion per row: the first non-pad id selects the text.
        return [table[int(row[0])] for row in ids.tolist()]

    module = BradleyTerryRewardModel(bt, decode)
    ids = torch.tensor([[1, 5, 0, 0], [2, 6, 7, 0]])
    scores = reward_at_last_token(module, ids, pad_id=0)
    direct = bt.reward([(prompt, good), (prompt, bad)])
    assert scores.tolist() == pytest.approx(direct.tolist())
    assert scores[0] > scores[1]


def test_ppo_training_and_dpo_both_raise_the_preferred_completion():
    _bt, groups = toy_groups()
    assert groups[0][0]["reward"] > groups[0][1]["reward"]
    log = []
    model = _tiny()
    train_ppo(model, groups, {"lora_r": 4, "lora_alpha": 8, "seed": 0, "epochs": 4, "lr": 5e-3, "clip": 0.2,
                              "kl_coef": 0.02}, log)
    assert log and all(math_ok(row) for row in log)
    result = compare_with_dpo(
        groups,
        {"lora_r": 4, "lora_alpha": 8, "seed": 0, "epochs": 12, "lr": 5e-3, "batch": 2, "beta": 0.5,
         "kl_coef": 0.02, "clip": 0.2},
        _tiny,
    )
    assert result["ppo_preferred_rose"] and result["dpo_preferred_rose"]
    assert result["dpo_final_pair_acc"] == 1.0


def math_ok(row: dict) -> bool:
    return row["loss"] == row["loss"] and row["kl"] >= 0
