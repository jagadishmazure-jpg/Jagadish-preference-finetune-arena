"""QLoRA supervised fine-tuning, and the numbers used to compare it with the other ways to tune.

The GPU job (``scripts/qlora_sft.py``, ``infra/aml/qlora-job.yml``) loads Phi-3.5-mini in 4-bit with
bitsandbytes and trains a PEFT adapter through TRL's ``SFTTrainer``. Nothing in CI has a GPU or those
libraries, so this module also has:

* NF4 plus double-quantisation of the scales, the two ideas QLoRA adds on top of LoRA, in plain PyTorch
  so a test can check them.
* A PEFT-formula adapter (``y = Wx + dropout(x) A B * alpha/r``, B starts at zero). With dropout 0 it
  matches ``prefarena.lora.LoRALinear``.
* A CPU smoke that quantises a tiny random model, dequantises it back to fp32 for the matmul (there is
  no bitsandbytes kernel on CPU in CI) and runs a few SFT steps.

VRAM, parameter and list-price figures below are arithmetic on published architecture numbers and
published prices. They are not measurements from a GPU run. The only measured accuracy in the
comparison is the committed SmolLM2-135M SFT rubric score.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# bitsandbytes NF4 codebook: 16 quantiles of a normal, scaled to [-1, 1]. The GPU job asks
# bitsandbytes for this same table via bnb_4bit_quant_type="nf4".
NF4_CODEBOOK = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)

# microsoft/Phi-3.5-mini-instruct config. params is the published safetensors count.
PHI35 = {
    "model_id": "microsoft/Phi-3.5-mini-instruct",
    "params": 3_821_079_552,
    "layers": 32,
    "hidden": 3072,
    "heads": 32,
}
LLAMA31_8B = {
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "params": 8_030_261_248,
    "layers": 32,
    "hidden": 4096,
    "heads": 32,
}

# Linux pay-as-you-go, eastus, Azure retail list checked 2026-11-17. Not a measured bill.
A100_SKU = "Standard_NC24ads_A100_v4"
H100_SKU = "Standard_NC40ads_H100_v5"
A100_USD_PER_HOUR = 3.673
H100_USD_PER_HOUR = 6.980
# Published Azure OpenAI GPT-4o mini fine-tune training rate: $0.0033 per 1K tokens ($3.30 / 1M).
# The catalogue changes; recheck Foundry pricing before submitting infra/aml/aoai-finetune.yml.
AOAI_USD_PER_MILLION_TOKENS = 3.30
AOAI_MODEL = "gpt-4o-mini"

# The CPU smoke is a tiny Llama, which has separate q, k and v projections.
# Phi-3.5-mini fuses those three into qkv_proj (hidden -> 3 * hidden) plus o_proj.
LLAMA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
PHI35_TARGETS = ("qkv_proj", "o_proj")
QLORA_TARGETS = LLAMA_TARGETS
QLORA_R = 16
QLORA_ALPHA = 32
QLORA_DROPOUT = 0.05


def lora_targets_for(model_id: str) -> tuple[str, ...]:
    if "phi" in model_id.lower():
        return PHI35_TARGETS
    return LLAMA_TARGETS


def lora_trainable(layers: int, hidden: int, r: int, matrices: int = 4) -> int:
    """A and B on each square attention projection: 2 * hidden * r per matrix."""
    return layers * matrices * 2 * hidden * r


def phi_attention_lora_params(layers: int = 32, hidden: int = 3072, r: int = 16) -> int:
    """Trainable count for Phi-3.5 attention LoRA: fused qkv_proj plus o_proj."""
    qkv = hidden * r + (3 * hidden) * r
    out_proj = hidden * r + hidden * r
    return layers * (qkv + out_proj)


def nf4_storage_bytes(n_params: int, block: int = 64, double_quant: bool = True, scale_block: int = 256) -> int:
    """Bytes to store ``n_params`` weights in NF4. Codes are 4 bits. Scales are fp32, or 8-bit with one
    fp32 absmax per ``scale_block`` scales when double quantisation is on (QLoRA's nested quant)."""
    if n_params < 0 or block < 1:
        raise ValueError("n_params and block must be positive")
    codes = (n_params + 1) // 2  # 4 bits each, packed
    n_blocks = math.ceil(n_params / block)
    if not double_quant:
        return codes + 4 * n_blocks
    n_scale_blocks = math.ceil(n_blocks / scale_block)
    return codes + n_blocks + 4 * n_scale_blocks  # 1 byte per scale + fp32 absmaxes


def weight_bytes(n_params: int, kind: str) -> int:
    if kind == "fp32":
        return n_params * 4
    if kind == "fp16":
        return n_params * 2
    if kind == "nf4":
        return nf4_storage_bytes(n_params)
    raise ValueError(kind)


def aoai_training_usd(tokens: int, epochs: int, usd_per_million: float = AOAI_USD_PER_MILLION_TOKENS) -> float:
    """Foundry bills training tokens times epochs times the per-token rate."""
    return tokens * epochs * usd_per_million / 1_000_000


def gpu_job_usd(hours: float, sku: str = A100_SKU) -> float:
    rate = {A100_SKU: A100_USD_PER_HOUR, H100_SKU: H100_USD_PER_HOUR}[sku]
    return hours * rate


def peft_lora_config_kwargs(r: int = QLORA_R, alpha: int = QLORA_ALPHA, dropout: float = QLORA_DROPOUT,
                            targets: tuple[str, ...] | None = None, model_id: str | None = None) -> dict:
    """Keyword arguments for ``peft.LoraConfig``. Kept here so CI can check them without installing PEFT.

    The default target list is Phi-3.5's fused attention projections. Pass ``LLAMA_TARGETS`` for a Llama.
    """
    chosen = targets if targets is not None else lora_targets_for(model_id or PHI35["model_id"])
    return {
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": list(chosen),
    }


def bitsandbytes_config_kwargs() -> dict:
    """Keyword arguments for ``transformers.BitsAndBytesConfig`` on the GPU job."""
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
    }


def build_peft_config(model_id: str | None = None):
    """Real ``peft.LoraConfig``. Raises ImportError when PEFT is not installed (the CPU smoke does not need it)."""
    from peft import LoraConfig

    return LoraConfig(**peft_lora_config_kwargs(model_id=model_id or PHI35["model_id"]))


def serving_quant_config():
    """4-bit load for the T4 deployment. T4 has no bfloat16, so the compute dtype is float16.

    Training on A100/H100 uses ``build_bitsandbytes_config`` (bfloat16) instead.
    """
    import torch as th
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=th.float16,
    )


def build_bitsandbytes_config():
    import torch as th
    from transformers import BitsAndBytesConfig

    kwargs = bitsandbytes_config_kwargs()
    kwargs["bnb_4bit_compute_dtype"] = th.bfloat16
    return BitsAndBytesConfig(**kwargs)


def build_sft_trainer(model_id: str, train_dataset, output_dir: str):
    """TRL ``SFTTrainer`` with 4-bit NF4 and a PEFT adapter. GPU only: bitsandbytes has no CPU kernel here."""
    from trl import SFTConfig, SFTTrainer

    args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        max_steps=-1,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_length=1024,
        dataset_text_field="text",
        report_to="none",
        seed=0,
    )
    return SFTTrainer(
        model=model_id,
        args=args,
        train_dataset=train_dataset,
        peft_config=build_peft_config(),
        quantization_config=build_bitsandbytes_config(),
    )


class PeftLoRALinear(nn.Module):
    """The PEFT LoRA forward, written out so it can be compared with ``LoRALinear`` without the library.

    PEFT computes ``result + dropout(x) @ A.T @ B.T * (alpha / r)`` with B initialised to zero.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale, self.enabled = r, alpha / r, True
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + (self.drop(x) @ self.A.T @ self.B.T) * self.scale
        return y


def apply_peft_lora(model: nn.Module, targets=QLORA_TARGETS, r: int = 8, alpha: int = 16,
                    dropout: float = 0.0) -> int:
    for p in model.parameters():
        p.requires_grad_(False)
    for _name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(module, child_name, PeftLoRALinear(child, r, alpha, dropout))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _codebook(device, dtype):
    return torch.tensor(NF4_CODEBOOK, device=device, dtype=dtype)


def quantize_nf4(weight: torch.Tensor, block: int = 64, double_quant: bool = True, scale_block: int = 256):
    """Block-wise NF4. Returns codes, reconstructed scales' ingredients, and the original shape.

    Codes are int64 indices into ``NF4_CODEBOOK`` (the packed 4-bit form is what ``nf4_storage_bytes``
    counts; the index tensor is only here so the CPU smoke can dequantise).
    """
    if weight.ndim == 0:
        raise ValueError("weight must have a shape")
    flat = weight.detach().float().reshape(-1)
    n = flat.numel()
    pad = (block - n % block) % block
    padded = torch.nn.functional.pad(flat, (0, pad)) if pad else flat
    blocks = padded.view(-1, block)
    scale = blocks.abs().amax(dim=1).clamp(min=1e-8)
    codes = (blocks / scale[:, None]).unsqueeze(-1).sub(_codebook(flat.device, flat.dtype)).abs().argmin(-1)
    scale_codes, scale_amax, scale_pad = None, None, 0
    if double_quant:
        scale_codes, scale_amax, scale_pad = _quantize_scales(scale, scale_block)
        scale = _dequant_scales(scale_codes, scale_amax, scale_pad, scale_block)
    return {
        "codes": codes,
        "scale": scale,
        "shape": tuple(weight.shape),
        "pad": pad,
        "block": block,
        "double_quant": double_quant,
        "scale_codes": scale_codes,
        "scale_amax": scale_amax,
        "scale_pad": scale_pad,
        "scale_block": scale_block,
        "storage_bytes": nf4_storage_bytes(n, block, double_quant, scale_block),
    }


def _quantize_scales(scale: torch.Tensor, block: int):
    n = scale.numel()
    pad = (block - n % block) % block
    padded = torch.nn.functional.pad(scale, (0, pad)) if pad else scale
    blocks = padded.view(-1, block)
    amax = blocks.amax(dim=1).clamp(min=1e-8)
    codes = torch.round(blocks / amax[:, None] * 255).clamp(0, 255).to(torch.int64)
    return codes, amax, pad


def _dequant_scales(codes: torch.Tensor, amax: torch.Tensor, pad: int, block: int) -> torch.Tensor:
    flat = (codes.float() / 255 * amax[:, None]).reshape(-1)
    if pad:
        flat = flat[:-pad]
    return flat


def dequantize_nf4(packed: dict) -> torch.Tensor:
    scale = packed["scale"]
    if packed["double_quant"]:
        scale = _dequant_scales(packed["scale_codes"], packed["scale_amax"], packed["scale_pad"], packed["scale_block"])
    values = _codebook(scale.device, scale.dtype)[packed["codes"]] * scale[:, None]
    flat = values.reshape(-1)
    if packed["pad"]:
        flat = flat[:-packed["pad"]]
    return flat.reshape(packed["shape"])


def quantize_linears_(model: nn.Module, block: int = 64) -> int:
    """Replace every plain linear's weight with its NF4 reconstruction and remember the storage size.

    The GPU job never does this: bitsandbytes keeps the weight in 4-bit and dequantises inside the
    matmul. On CPU the reconstructed fp32 weight is what the matmul sees, and ``storage_bytes`` is
    what 4-bit would have stored.
    """
    total = 0
    for module in model.modules():
        # LoRA wrappers are not nn.Linear, so this only sees the frozen base projections.
        if isinstance(module, nn.Linear):
            packed = quantize_nf4(module.weight.data, block=block)
            with torch.no_grad():
                module.weight.copy_(dequantize_nf4(packed).to(dtype=module.weight.dtype))
            module._nf4_storage_bytes = packed["storage_bytes"]  # noqa: SLF001
            total += packed["storage_bytes"]
    return total


def nf4_bytes_of(model: nn.Module) -> int:
    return sum(getattr(m, "_nf4_storage_bytes", 0) for m in model.modules())


def apply_qlora(model: nn.Module, targets=QLORA_TARGETS, r: int = 8, alpha: int = 16, block: int = 64) -> int:
    """Quantise the frozen base, then attach PEFT-formula adapters. Returns trainable parameter count."""
    quantize_linears_(model, block=block)
    return apply_peft_lora(model, targets=targets, r=r, alpha=alpha, dropout=0.0)


def smoke_flags(epochs: int = 8) -> dict:
    """Train hand-written LoRA, PEFT-formula LoRA and NF4 QLoRA on one tiny random model. CPU, no download."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from .tuning import train_sft

    def tiny():
        torch.manual_seed(0)
        cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
        model = LlamaForCausalLM(cfg)
        model.eval()
        return model

    examples = [([1, 2, 3, 4], [5, 6, 7, 8]), ([2, 3, 4, 5], [6, 7, 8, 9]),
                ([3, 1, 4, 2], [8, 7, 6, 5]), ([4, 4, 1, 2], [9, 8, 7, 6])]
    cfg = {"lora_r": 4, "lora_alpha": 8, "seed": 0, "epochs": epochs, "lr": 5e-3, "batch": 2}
    pad = type("T", (), {"pad_token_id": 0})()

    def run(apply):
        model = tiny()
        log: list[dict] = []
        train_sft(model, pad, examples, cfg, log, apply=apply)
        return log[0]["loss"], log[-1]["loss"]

    def peft_apply(model, r, alpha):
        return apply_peft_lora(model, r=r, alpha=alpha, dropout=0.0)

    def qlora_apply(model, r, alpha):
        return apply_qlora(model, r=r, alpha=alpha)

    h0, h1 = run(None)
    p0, p1 = run(peft_apply)
    q0, q1 = run(qlora_apply)
    return {
        "handwritten_loss_fell": h1 < h0,
        "peft_loss_fell": p1 < p0,
        "qlora_loss_fell": q1 < q0,
        "peft_matches_handwritten": abs(h0 - p0) < 1e-4 and abs(h1 - p1) < 1e-4,
    }


def supervised_messages(prompt: dict, response: str) -> list[dict]:
    """Chat messages for one SFT example: deploy system prompt, policy context, then the chosen answer."""
    from .styles import DEPLOY_SYSTEM, messages

    return messages(prompt, DEPLOY_SYSTEM) + [{"role": "assistant", "content": response}]


def serving_messages(user_content: str) -> list[dict]:
    """The same system prompt and user turn the adapter was trained on, without the answer yet.

    ``user_content`` must be ``styles.user_message(prompt)`` (policy plus the customer question).
    The scoring script sends these messages through the model's chat template.
    """
    from .styles import DEPLOY_SYSTEM

    return [{"role": "system", "content": DEPLOY_SYSTEM}, {"role": "user", "content": user_content}]


def production_rows() -> list[dict]:
    """One row per method. Accuracy is measured only for the committed CPU SFT run; the caller fills that in."""
    phi_lora = lora_trainable(PHI35["layers"], PHI35["hidden"], QLORA_R)
    rows = [
        {
            "method": "handwritten_lora",
            "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "where": "laptop CPU, measured",
            "weight_bytes": None,
            "trainable": 921_600,
            "accuracy": "see committed SFT rubric score",
        },
        {
            "method": "peft_lora",
            "model": PHI35["model_id"],
            "where": "GPU job written, not run",
            "weight_bytes": weight_bytes(PHI35["params"], "fp16"),
            "trainable": phi_lora,
            "accuracy": "not measured",
        },
        {
            "method": "qlora",
            "model": PHI35["model_id"],
            "where": "GPU job written, not run",
            "weight_bytes": weight_bytes(PHI35["params"], "nf4"),
            "trainable": phi_lora,
            "accuracy": "not measured",
        },
        {
            "method": "azure_openai",
            "model": AOAI_MODEL,
            "where": "Foundry fine-tune written, not run",
            "weight_bytes": None,
            "trainable": None,
            "accuracy": "not measured",
        },
    ]
    return rows
