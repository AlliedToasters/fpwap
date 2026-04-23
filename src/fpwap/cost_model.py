"""Cost model for predicting fpwap run wall-clock and throughput.

Pure arithmetic — no GPU, no model loading. Fed by measurements from the
preflight single-layer probe, returns a predicted timing breakdown and
bottleneck classification.

Cost model per layer:
    compute_per_layer = fwd_per_microbatch * ceil(n_samples / microbatch_size)
    load_per_layer    = weight_load_s

    With prefetch:    per_layer = max(load, compute)
    Without prefetch: per_layer = load + compute

    total = embed_s + per_layer * n_layers

References: FlexGen §4.3 (Sheng et al., ICML 2023), fpwap SPEC §10.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CostModelInput:
    """Measured quantities from a single-layer preflight probe."""

    n_layers: int
    n_samples: int
    seq_len: int
    microbatch_size: int
    weight_load_s: float
    fwd_per_microbatch_s: float
    embed_s: float
    layer_weight_bytes: int


@dataclass(frozen=True)
class CostModelPrediction:
    """Predicted run timing for a specific configuration."""

    per_layer_s: float
    total_wall_s: float
    throughput_tok_s: float
    bottleneck: str
    load_pct: float
    compute_pct: float
    weight_io_gb: float
    prefetch: bool


def predict(inp: CostModelInput, *, prefetch: bool) -> CostModelPrediction:
    """Predict end-to-end wall-clock from preflight probe measurements."""
    n_microbatches = -(-inp.n_samples // inp.microbatch_size)

    compute_per_layer = inp.fwd_per_microbatch_s * n_microbatches
    load_per_layer = inp.weight_load_s

    if prefetch and load_per_layer > 0:
        per_layer = max(load_per_layer, compute_per_layer)
    else:
        per_layer = load_per_layer + compute_per_layer

    total = inp.embed_s + per_layer * inp.n_layers
    total_tokens = inp.n_samples * inp.seq_len
    throughput = total_tokens / total if total > 0.0 else 0.0

    non_overlap = load_per_layer + compute_per_layer
    if non_overlap > 0:
        load_pct = load_per_layer / non_overlap
        compute_pct = compute_per_layer / non_overlap
    else:
        load_pct = 0.0
        compute_pct = 0.0

    if load_pct > 0.55:
        bottleneck = "load"
    elif compute_pct > 0.55:
        bottleneck = "compute"
    else:
        bottleneck = "balanced"

    weight_io_gb = inp.layer_weight_bytes * inp.n_layers / 1e9

    return CostModelPrediction(
        per_layer_s=per_layer,
        total_wall_s=total,
        throughput_tok_s=throughput,
        bottleneck=bottleneck,
        load_pct=load_pct,
        compute_pct=compute_pct,
        weight_io_gb=weight_io_gb,
        prefetch=prefetch,
    )


@dataclass(frozen=True)
class CandidateConfig:
    """One configuration to evaluate in the parameter search."""

    cost_input: CostModelInput
    buffer_device: str
    prefetch: bool


@dataclass(frozen=True)
class Recommendation:
    """Best configuration found by the cost model search."""

    microbatch_size: int
    buffer_device: str
    prefetch: bool
    prediction: CostModelPrediction


def recommend(candidates: Sequence[CandidateConfig]) -> Recommendation:
    """Evaluate candidate configs and return the highest-throughput one."""
    if not candidates:
        raise ValueError("no candidate configurations to evaluate")

    best: Recommendation | None = None
    for c in candidates:
        pred = predict(c.cost_input, prefetch=c.prefetch)
        if best is None or pred.throughput_tok_s > best.prediction.throughput_tok_s:
            best = Recommendation(
                microbatch_size=c.cost_input.microbatch_size,
                buffer_device=c.buffer_device,
                prefetch=c.prefetch,
                prediction=pred,
            )

    assert best is not None
    return best
