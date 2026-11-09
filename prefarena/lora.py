"""Minimal LoRA for PyTorch linear layers: y = W x + (B A x) * alpha / r, with W frozen.

Small enough to read in one go. `set_adapter(model, False)` turns every adapter off, which gives the
frozen reference model DPO needs without a second copy in memory.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale, self.enabled = r, alpha / r, True
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # zero: starts as an exact no-op
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + (x @ self.A.T @ self.B.T) * self.scale
        return y

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        lin.weight.copy_(self.base.weight + (self.B @ self.A) * self.scale)
        if self.base.bias is not None:
            lin.bias.copy_(self.base.bias)
        return lin


def apply_lora(model: nn.Module, targets=("q_proj", "k_proj", "v_proj", "o_proj"), r: int = 8, alpha: int = 16) -> int:
    """Freeze the model and wrap target linears. Returns the number of trainable parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    for _name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r, alpha))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_adapter(model: nn.Module, enabled: bool) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = enabled


def merge_lora(model: nn.Module) -> None:
    """Fold adapters into plain linears (used to start DPO from the SFT model)."""
    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, child_name, child.merged())


def lora_state(model: nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k.endswith((".A", ".B"))}
