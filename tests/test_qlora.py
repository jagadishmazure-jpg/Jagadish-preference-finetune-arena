"""QLoRA pieces that run on CPU: NF4, the PEFT formula, the smoke, and the job YAML."""

import math

import pytest

torch = pytest.importorskip("torch")

from conftest import ROOT  # noqa: E402

from prefarena.lora import LoRALinear  # noqa: E402
from prefarena.qlora import (  # noqa: E402
    A100_SKU,
    H100_SKU,
    PHI35,
    PeftLoRALinear,
    aoai_training_usd,
    apply_peft_lora,
    bitsandbytes_config_kwargs,
    dequantize_nf4,
    lora_trainable,
    nf4_storage_bytes,
    peft_lora_config_kwargs,
    quantize_nf4,
    smoke_flags,
    weight_bytes,
)
from prefarena.tuning import train_sft  # noqa: E402


def test_nf4_roundtrip_is_close_and_double_quant_uses_fewer_bytes():
    torch.manual_seed(0)
    weight = torch.randn(128, 64)
    plain = quantize_nf4(weight, double_quant=False)
    nested = quantize_nf4(weight, double_quant=True)
    for packed in (plain, nested):
        err = (dequantize_nf4(packed) - weight).abs()
        assert float(err.mean()) < 0.1 and float(err.max()) < 0.5
    assert nested["storage_bytes"] < plain["storage_bytes"]
    assert nf4_storage_bytes(weight.numel(), double_quant=True) == nested["storage_bytes"]
    assert weight_bytes(PHI35["params"], "nf4") < weight_bytes(PHI35["params"], "fp16") < weight_bytes(PHI35["params"], "fp32")


def test_peft_formula_matches_handwritten_lora_when_dropout_is_zero():
    torch.manual_seed(0)
    base = torch.nn.Linear(8, 8)
    torch.manual_seed(1)
    handwritten = LoRALinear(base, r=4, alpha=8)
    torch.manual_seed(1)
    peft = PeftLoRALinear(base, r=4, alpha=8, dropout=0.0)
    x = torch.randn(2, 3, 8)
    assert torch.allclose(handwritten(x), peft(x))
    with torch.no_grad():
        handwritten.B.normal_()
        peft.B.copy_(handwritten.B)
    assert torch.allclose(handwritten(x), peft(x))


def test_qlora_adapter_is_a_no_op_until_b_moves_and_the_smoke_learns():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32)
    model = LlamaForCausalLM(cfg)
    before = model(torch.tensor([[1, 2, 3, 4]])).logits
    from prefarena.qlora import quantize_linears_

    quantize_linears_(model)
    changed = model(torch.tensor([[1, 2, 3, 4]])).logits
    assert not torch.allclose(before, changed)
    frozen = changed.detach().clone()
    apply_peft_lora(model, r=2, alpha=4, dropout=0.0)
    assert torch.allclose(frozen, model(torch.tensor([[1, 2, 3, 4]])).logits)
    flags = smoke_flags(epochs=6)
    assert flags["handwritten_loss_fell"] and flags["peft_matches_handwritten"] and flags["qlora_loss_fell"]
    assert flags["peft_loss_fell"]


def test_phi35_lora_count_and_aoai_bill_are_arithmetic():
    assert lora_trainable(32, 3072, 16) == 32 * 4 * 2 * 3072 * 16
    assert aoai_training_usd(1_000_000, 3) == pytest.approx(9.9)
    assert math.isclose(aoai_training_usd(0, 3), 0.0)


def test_library_configs_describe_nf4_and_the_attention_projections():
    bnb = bitsandbytes_config_kwargs()
    assert bnb["load_in_4bit"] and bnb["bnb_4bit_quant_type"] == "nf4" and bnb["bnb_4bit_use_double_quant"]
    peft = peft_lora_config_kwargs()
    assert peft["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert peft["task_type"] == "CAUSAL_LM"


def test_gpu_job_yamls_name_the_skus_the_libraries_and_the_endpoint():
    job = (ROOT / "infra/aml/qlora-job.yml").read_text()
    deploy = (ROOT / "infra/aml/qlora-deployment.yml").read_text()
    endpoint = (ROOT / "infra/aml/qlora-endpoint.yml").read_text()
    env = (ROOT / "infra/aml/environment-gpu.yml").read_text()
    ppo = (ROOT / "infra/aml/ppo-job.yml").read_text()
    ppo_env = (ROOT / "infra/aml/environment-ppo.yml").read_text()
    assert A100_SKU in job and H100_SKU in job and "scripts/qlora_sft.py" in job
    assert "peft==" in env and "trl==" in env and "bitsandbytes==" in env
    assert "aad_token" in endpoint and "scripts/score_qlora.py" in deploy
    assert "phi35-qlora-adapter" in deploy and "phi35-qlora-adapter" in (ROOT / "infra/aml/qlora-model.yml").read_text()
    assert "scripts/ppo_trl.py" in ppo and "trl[peft]==1.11.0" in ppo_env
    assert "not an Azure ML command job" in (ROOT / "infra/aml/aoai-finetune.yml").read_text() or "not an Azure ML" in (
        ROOT / "infra/aml/aoai-finetune.yml").read_text()


def test_sft_loop_accepts_the_peft_apply_hook():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32)
    model = LlamaForCausalLM(cfg)
    log = []
    train_sft(model, type("T", (), {"pad_token_id": 0})(), [([1, 2], [3, 4])],
              {"lora_r": 2, "lora_alpha": 4, "seed": 0, "epochs": 2, "lr": 1e-2, "batch": 1}, log,
              apply=lambda m, r, alpha: apply_peft_lora(m, r=r, alpha=alpha))
    assert log[-1]["loss"] <= log[0]["loss"]
