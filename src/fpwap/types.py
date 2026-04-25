from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import torch
from torch import Tensor

HookName = Literal["residual_pre", "residual_post", "attn_out", "mlp_out"]
LoadingStrategy = Literal["cpu_offload", "disk_offload", "mmap_from_cache"]
PaddingMode = Literal["fixed", "bucketed"]
Phase = Literal["read", "write", "read_after_write"]


@dataclass(frozen=True)
class Emit:
    tensor: Tensor
    dtype: torch.dtype | None = None
    sample_lengths: Tensor | None = None


@dataclass(frozen=True)
class WriteBack:
    tensor: Tensor


BatchResult: TypeAlias = Emit | WriteBack | None


@dataclass(frozen=True)
class RaggedTensor:
    """Variable-length per-sample tensor, stored as a flat tensor + offsets.

    `flat` is `[total_tokens, *trailing_shape]`. `offsets` is `[N_samples + 1]`
    int64; sample i lives at `flat[offsets[i]:offsets[i+1]]`. Returned by
    `Result.activations(...)` and `MemmapBackend.read_all(...)` when the
    underlying emit was ragged (callback supplied `Emit.sample_lengths`).
    """

    flat: Tensor
    offsets: Tensor

    def __len__(self) -> int:
        return int(self.offsets.shape[0]) - 1

    def __getitem__(self, i: int) -> Tensor:
        start = int(self.offsets[i].item())
        stop = int(self.offsets[i + 1].item())
        return self.flat[start:stop]

    @property
    def lengths(self) -> Tensor:
        return self.offsets[1:] - self.offsets[:-1]


@dataclass(frozen=True)
class ArtifactKey:
    sweep_id: str
    layer_idx: int
    hook: HookName
    kind: str


@dataclass
class LayerArtifact:
    kind: str
    payload: Any


@dataclass
class Artifact:
    key: ArtifactKey
    payload: Any


@dataclass
class Context:
    sweep_id: str
    n_samples: int
    seq_len: int
    hidden: int
    transport_dtype: torch.dtype
