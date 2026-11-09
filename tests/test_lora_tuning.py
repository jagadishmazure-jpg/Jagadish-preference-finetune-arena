"""LoRA and the training loops, on a tiny randomly initialised Llama (no download, a few seconds on CPU)."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from prefarena import tuning  # noqa: E402
from prefarena.lora import LoRALinear, apply_lora, merge_lora, set_adapter  # noqa: E402

TOK = SimpleNamespace(pad_token_id=0)


def tiny():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    m = transformers.LlamaForCausalLM(cfg)
    m.eval()
    return m


IDS = torch.tensor([[5, 6, 7, 8, 9, 10]])


def test_lora_starts_as_an_exact_no_op_and_trains_only_adapters():
    m = tiny()
    before = m(IDS).logits
    n = apply_lora(m, r=4, alpha=8)
    assert torch.allclose(before, m(IDS).logits)
    trainable = {k for k, p in m.named_parameters() if p.requires_grad}
    assert trainable and all(k.endswith((".A", ".B")) for k in trainable)
    assert n == sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert sum(isinstance(x, LoRALinear) for x in m.modules()) == 2 * 4  # q, k, v, o in each of 2 layers


def test_switching_adapters_off_gives_back_the_reference_model():
    m = tiny()
    ref = m(IDS).logits
    apply_lora(m, r=4, alpha=8)
    for mod in m.modules():
        if isinstance(mod, LoRALinear):
            torch.nn.init.normal_(mod.B, std=0.5)
    assert not torch.allclose(ref, m(IDS).logits)
    set_adapter(m, False)
    assert torch.allclose(ref, m(IDS).logits)


def test_merging_keeps_the_adapted_output():
    m = tiny()
    apply_lora(m, r=4, alpha=8)
    for mod in m.modules():
        if isinstance(mod, LoRALinear):
            torch.nn.init.normal_(mod.B, std=0.1)
    adapted = m(IDS).logits
    merge_lora(m)
    assert not any(isinstance(x, LoRALinear) for x in m.modules())
    assert torch.allclose(adapted, m(IDS).logits, atol=1e-5)


def test_sequence_log_probs_count_only_response_tokens():
    m = tiny()
    a, _ = tuning.seq_logps(m, [([1, 2, 3], [4, 5])], 0)
    b, _ = tuning.seq_logps(m, [([1, 2, 3], [4, 5]), ([1], [4, 5, 6, 7, 8])], 0)  # padding must not change row 0
    assert torch.allclose(a, b[:1], atol=1e-5) and a.item() < 0


def test_dpo_training_raises_the_chosen_margin():
    m = tiny()
    pairs = [([1, 2, 3], [10, 11, 12], [20, 21, 22]), ([1, 4], [10, 11], [30, 31])]
    log = []
    tuning.train_pref(m, TOK, pairs, {"lora_r": 4, "lora_alpha": 8, "seed": 0, "epochs": 15, "lr": 5e-3, "batch": 2,
                                      "beta": 0.5}, log, "dpo")
    assert log[0]["loss"] == pytest.approx(0.6931, abs=1e-3)  # policy == reference at the start
    assert log[-1]["loss"] < log[0]["loss"] and log[-1]["acc"] == 1.0


def test_orpo_and_sft_reduce_their_losses():
    pairs = [([1, 2, 3], [10, 11, 12], [20, 21, 22])]
    m, log = tiny(), []
    tuning.train_pref(m, TOK, pairs, {"lora_r": 4, "lora_alpha": 8, "seed": 0, "epochs": 10, "lr": 5e-3, "batch": 1,
                                      "lam": 0.5}, log, "orpo")
    assert log[-1]["loss"] < log[0]["loss"]
    m2, log2 = tiny(), []
    tuning.train_sft(m2, TOK, [([1, 2], [10, 11, 12])], {"lora_r": 4, "lora_alpha": 8, "seed": 0, "epochs": 10,
                                                         "lr": 5e-3, "batch": 1}, log2)
    assert log2[-1]["loss"] < log2[0]["loss"]


def test_losses_give_the_same_numbers_in_torch_and_numpy():
    import numpy as np

    from prefarena.losses import dpo_loss

    args = ([-3.0], [-9.0], [-4.0], [-6.0])
    t, _ = dpo_loss(torch, *(torch.tensor(a) for a in args), beta=0.3)
    n, _ = dpo_loss(np, *(np.array(a) for a in args), beta=0.3)
    assert t.item() == pytest.approx(float(n[0]))
