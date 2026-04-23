from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import torch
from torch import Tensor

from fpwap.callbacks.base import fpwapCallback
from fpwap.types import (
    BatchResult,
    HookName,
    LayerArtifact,
    fpwapArtifact,
)


class RawActivations(fpwapCallback):
    phase = "read"

    def __init__(
        self,
        layers: Sequence[int] | Literal["all"] = "all",
        hook: HookName = "residual_post",
        last_token_only: bool = True,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.target_layers = layers
        self.target_hooks = (hook,)
        self.last_token_only = last_token_only
        self.out_dtype = out_dtype

    def on_batch(
        self,
        layer_idx: int,
        hook: HookName,
        acts: Tensor,
        sample_ids: Tensor,
    ) -> BatchResult:
        raise NotImplementedError


class IncrementalPCA(fpwapCallback):
    phase = "read"

    def __init__(
        self,
        layers: Sequence[int] | Literal["all"] = "all",
        n_components: int = 64,
        hook: HookName = "residual_post",
    ) -> None:
        self.target_layers = layers
        self.target_hooks = (hook,)
        self.n_components = n_components

    def on_batch(
        self,
        layer_idx: int,
        hook: HookName,
        acts: Tensor,
        sample_ids: Tensor,
    ) -> BatchResult:
        raise NotImplementedError

    def on_fpwap_end(self) -> fpwapArtifact | None:
        raise NotImplementedError


class DiffOfMeans(fpwapCallback):
    phase = "read"

    def __init__(
        self,
        layers: Sequence[int] | Literal["all"] = "all",
        label_fn: Callable[[Tensor], Tensor] | None = None,
        hook: HookName = "residual_post",
    ) -> None:
        self.target_layers = layers
        self.target_hooks = (hook,)
        self.label_fn = label_fn

    def on_batch(
        self,
        layer_idx: int,
        hook: HookName,
        acts: Tensor,
        sample_ids: Tensor,
    ) -> BatchResult:
        raise NotImplementedError

    def on_layer_end(self, layer_idx: int) -> LayerArtifact | None:
        raise NotImplementedError


class SteerInBasis(fpwapCallback):
    phase = "write"

    def __init__(
        self,
        basis_artifact: fpwapArtifact,
        direction_idx: int,
        alpha: float,
        layers: Sequence[int] | Literal["all"] = "all",
        hook: HookName = "residual_post",
    ) -> None:
        self.target_layers = layers
        self.target_hooks = (hook,)
        self.basis = basis_artifact
        self.direction_idx = direction_idx
        self.alpha = alpha

    def on_batch(
        self,
        layer_idx: int,
        hook: HookName,
        acts: Tensor,
        sample_ids: Tensor,
    ) -> BatchResult:
        raise NotImplementedError
