from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from fpwap.callbacks.base import Callback
from fpwap.cost_model import CostModelPrediction
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
    prediction: CostModelPrediction | None = None
    recommended_prefetch: bool | None = None
    recommended_buffer_device: str | None = None

    def summary(self) -> str:
        lines: list[str] = []
        if self.feasible:
            lines.append("feasible")
        else:
            lines.append("not feasible")
            for b in self.blockers:
                lines.append(f"  blocker: {b}")
            return "\n".join(lines)

        lines.append(
            f"  microbatch_size={self.microbatch_size}  "
            f"residual_buffer={self.residual_buffer_gb:.2f} GB  "
            f"weight_io={self.estimated_weight_io_gb:.1f} GB"
        )

        if self.prediction is not None:
            p = self.prediction
            lines.append(
                f"  predicted throughput {p.throughput_tok_s:,.1f} tok/s  "
                f"wall {p.total_wall_s:.1f}s  "
                f"bottleneck={p.bottleneck}"
            )
            lines.append(
                f"  load {p.load_pct:.0%}  compute {p.compute_pct:.0%}"
            )
        else:
            lines.append(f"  estimated wall {self.estimated_wall_clock_s:.1f}s")

        if self.recommended_prefetch is not None:
            lines.append(f"  recommended: prefetch={self.recommended_prefetch}")
        if self.recommended_buffer_device is not None:
            lines.append(
                f"  recommended: buffer_device={self.recommended_buffer_device}"
            )

        for w in self.warnings:
            lines.append(f"  warning: {w}")

        return "\n".join(lines)


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
    callbacks: Sequence[Callback],
    transport_dtype: torch.dtype = torch.bfloat16,
) -> PreflightReport:
    raise NotImplementedError
