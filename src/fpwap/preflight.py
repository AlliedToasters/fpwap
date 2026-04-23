from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from fpwap.callbacks.base import fpwapCallback
from fpwap.types import LoadingStrategy


@dataclass
class PreflightReport:
    feasible: bool
    microbatch_size: int
    residual_buffer_gb: float
    per_layer_peak_vram_gb: float
    estimated_wall_clock_s: float
    estimated_weight_io_gb: float
    loading_strategy: LoadingStrategy
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _select_loading_strategy(
    model_size_bytes: int,
    cpu_ram_bytes: int,
    nvme_free_bytes: int,
) -> LoadingStrategy:
    """SPEC.md §12.2 decision tree. Pure arithmetic — the red->green demo."""
    if model_size_bytes <= int(cpu_ram_bytes * 0.7):
        return "cpu_offload"
    if (
        model_size_bytes <= int(cpu_ram_bytes * 1.5)
        and nvme_free_bytes > 2 * model_size_bytes
    ):
        return "disk_offload"
    return "mmap_from_cache"


def plan(
    model_spec: Any,
    dataset_size: int,
    seq_len: int,
    vram_budget_gb: float,
    nvme_free_gb: float,
    cpu_ram_gb: float,
    callbacks: Sequence[fpwapCallback],
    transport_dtype: torch.dtype = torch.bfloat16,
) -> PreflightReport:
    raise NotImplementedError
