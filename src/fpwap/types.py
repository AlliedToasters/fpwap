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


@dataclass(frozen=True)
class WriteBack:
    tensor: Tensor


BatchResult: TypeAlias = Emit | WriteBack | None


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
