from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

_SAFE_TO_TORCH_DTYPE: dict[str, str] = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}


def build_accel_index_from_hf_cache(snapshot_dir: Path) -> dict[str, dict[str, Any]]:
    """Convert HF's model.safetensors.index.json to accelerate's loader format."""
    raise NotImplementedError


def alias_tied_weights_in_index(
    model: nn.Module,
    accel_index: dict[str, dict[str, Any]],
) -> None:
    """Add aliases for tied weights to the index. Requires model.tie_weights() first."""
    raise NotImplementedError


def load_from_cache(
    model_id: str,
    snapshot_dir: Path,
    offload_dir: Path,
    execution_device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Load a model larger than CPU RAM via mmap-from-HF-cache."""
    raise NotImplementedError


def _load_layer(model: nn.Module, layer_idx: int, loader: Any) -> None:
    """Materialize layer `layer_idx` weights onto the execution device."""
    raise NotImplementedError


def _unload_layer(model: nn.Module, layer_idx: int) -> None:
    """Release layer `layer_idx` weights back to the meta device."""
    raise NotImplementedError
