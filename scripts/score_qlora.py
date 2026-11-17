"""Azure ML scoring script for the Phi-3.5 QLoRA adapter.

``init`` loads the base model and the registered adapter from ``AZUREML_MODEL_DIR``. ``run`` generates
one reply. The request parser is pure and is what CI tests; loading the weights needs the GPU image.
"""

from __future__ import annotations

import json
import os

_model = None
_tok = None
MAX_PROMPT_CHARS = 4000
MAX_NEW_TOKENS = 256


def parse_request(raw: str) -> dict:
    if not isinstance(raw, str):
        raise TypeError("request must be a JSON string")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("request is not JSON") from exc
    if not isinstance(data, dict) or "prompt" not in data:
        raise ValueError("prompt is required")
    prompt = str(data["prompt"]).strip()
    if not prompt:
        raise ValueError("prompt is empty")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError("prompt is too long")
    n = int(data.get("max_new_tokens", 128))
    if n < 1 or n > MAX_NEW_TOKENS:
        raise ValueError("max_new_tokens is out of range")
    return {"prompt": prompt, "max_new_tokens": n}


def _import_repo():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def init():
    global _model, _tok
    _import_repo()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from prefarena.qlora import serving_quant_config

    model_dir = os.environ["AZUREML_MODEL_DIR"]
    base = os.environ.get("BASE_MODEL", "microsoft/Phi-3.5-mini-instruct")
    _tok = AutoTokenizer.from_pretrained(base)
    if _tok.pad_token_id is None:
        _tok.pad_token = _tok.eos_token
    # NF4 weights, float16 compute: the T4 deployment has no bfloat16.
    base_model = AutoModelForCausalLM.from_pretrained(
        base, device_map="auto", quantization_config=serving_quant_config())
    _model = PeftModel.from_pretrained(base_model, model_dir)
    _model.eval()


def run(raw: str) -> str:
    request = parse_request(raw)
    if _model is None or _tok is None:
        raise RuntimeError("init was not called")
    _import_repo()
    import torch

    from prefarena.qlora import serving_messages

    # prompt is the training user turn: policy text plus the customer question.
    messages = serving_messages(request["prompt"])
    ids = _tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    ids = ids.to(_model.device)
    with torch.no_grad():
        out = _model.generate(ids, max_new_tokens=request["max_new_tokens"], do_sample=False)
    text = _tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True).strip()
    return json.dumps({"response": text})
