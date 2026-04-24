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
    recommended_buffer_device: str | None = None
    recommended_prefetch: bool | None = None

    def summary(self) -> str:
        """Human-readable preflight summary with cost model predictions."""
        lines: list[str] = []
        status = "feasible" if self.feasible else "INFEASIBLE"
        lines.append(f"{status} | microbatch_size={self.microbatch_size}")

        if self.blockers:
            for b in self.blockers:
                lines.append(f"  BLOCKER: {b}")

        if self.prediction is not None:
            p = self.prediction
            mins = p.total_wall_s / 60
            if mins >= 1.0:
                lines.append(
                    f"predicted: {mins:.1f} min wall-clock, "
                    f"{p.throughput_tok_s:,.0f} tok/s"
                )
            else:
                lines.append(
                    f"predicted: {p.total_wall_s:.1f}s wall-clock, "
                    f"{p.throughput_tok_s:,.0f} tok/s"
                )
            lines.append(
                f"bottleneck: {p.bottleneck} "
                f"(load {p.load_pct:.0%}, compute {p.compute_pct:.0%})"
            )
            lines.append(f"weight I/O: {p.weight_io_gb:.1f} GB")
        else:
            lines.append(
                f"estimated wall-clock: {self.estimated_wall_clock_s:.1f}s"
            )
            lines.append(f"weight I/O: {self.estimated_weight_io_gb:.1f} GB")

        if self.per_layer_peak_vram_gb > 0:
            lines.append(f"peak VRAM: {self.per_layer_peak_vram_gb:.1f} GB")

        recs: list[str] = []
        if self.recommended_buffer_device is not None:
            recs.append(f"buffer_device={self.recommended_buffer_device!r}")
        if self.recommended_prefetch is not None:
            recs.append(f"prefetch={'on' if self.recommended_prefetch else 'off'}")
        if recs:
            lines.append(f"recommended: {', '.join(recs)}")

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
