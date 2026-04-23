from __future__ import annotations

from torch import nn

from fpwap.models.base import ModelPlumbing
from fpwap.models.gpt2 import GPT2Plumbing
from fpwap.models.llama import LlamaPlumbing

_PLUMBING: list[ModelPlumbing] = [GPT2Plumbing(), LlamaPlumbing()]


def get_plumbing(model: nn.Module) -> ModelPlumbing:
    for p in _PLUMBING:
        if p.matches(model):
            return p
    raise NotImplementedError(
        f"no fpwap.models plumbing registered for {type(model).__name__}"
    )


__all__ = ["GPT2Plumbing", "LlamaPlumbing", "ModelPlumbing", "get_plumbing"]
