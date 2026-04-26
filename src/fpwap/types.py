from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


@dataclass(frozen=True)
class ResultArtifact:
    """On-disk handle returned by `Result.activations(..., as_path=True)`.

    Lets a caller skip materializing the emit into a host-RAM tensor when
    the consumer is going to read from disk anyway. The data file is
    backend-owned (in-place mode) or hardlinked/copied into a user dir
    (dest mode); ownership is never transferred.

    Fields:
        data_path:    The shard `.bin` file. mmap-readable.
        sidecar_path: The `.json` sidecar with shape/dtype (and per-sample
                      offsets for ragged). None only if a backend chose
                      not to write one.
        layout:       "dense" → `data_path` is `[N_samples, *per_row_shape]`.
                      "ragged" → flat `[total_tokens, *trailing]`; per-sample
                      offsets live in the sidecar.
        dtype:        Logical torch dtype of the on-disk tensor. bf16 is
                      stored as uint16 on disk; the sidecar `bf16_as_u16`
                      flag tells you to `.view(torch.bfloat16)` after read.
        shape:        Full tensor shape for dense; None for ragged (use
                      sidecar offsets / `RaggedTensor.lengths`).
    """

    data_path: Path
    sidecar_path: Path | None
    layout: Literal["dense", "ragged"]
    dtype: torch.dtype
    shape: tuple[int, ...] | None
