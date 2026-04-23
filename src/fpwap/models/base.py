from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from torch import Tensor, nn


class ModelPlumbing(Protocol):
    """Per-family hook plumbing. Implementations live in fpwap/models/<family>.py."""

    def layer_modules(self, model: nn.Module) -> Sequence[nn.Module]: ...

    def embed(self, model: nn.Module, input_ids: Tensor) -> Tensor: ...

    def construct_attention_mask(
        self,
        model: nn.Module,
        input_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor: ...
