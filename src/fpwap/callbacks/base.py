from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

from fpwap.types import (
    Artifact,
    BatchResult,
    Context,
    HookName,
    LayerArtifact,
    Phase,
)


class Callback:
    target_layers: Sequence[int] | Literal["all"] = "all"
    target_hooks: Sequence[HookName] = ("residual_post",)
    phase: Phase = "read"
    needs_grad: bool = False
    accum_dtype: torch.dtype = torch.float32

    def on_sweep_start(self, ctx: Context) -> None:
        return None

    def on_layer_start(self, layer_idx: int) -> None:
        return None

    def on_batch(
        self,
        layer_idx: int,
        hook: HookName,
        acts: Tensor,
        sample_ids: Tensor,
    ) -> BatchResult:
        return None

    def on_layer_end(self, layer_idx: int) -> LayerArtifact | None:
        return None

    def on_sweep_end(self) -> Artifact | None:
        return None

    def checkpoint_state(self) -> bytes:
        return b""

    def restore_state(self, state: bytes) -> None:
        return None
