"""Preflight cost model: predict per-layer latency and recommend config.

Pure arithmetic — no GPU, no model weights, no torch. CI-safe.

The cost model per layer:
  compute = fwd_per_microbatch_s × ceil(n_samples / microbatch_size)
  load    = weight_load_s

Without prefetch: per_layer = load + compute
With prefetch:    per_layer = max(load, compute)

Total wall = embed_s + per_layer × n_layers
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CostModelInput:
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
    per_layer_s: float
    total_wall_s: float
    throughput_tok_s: float
    bottleneck: str  # "load", "compute", or "balanced"
    load_pct: float
    compute_pct: float
    weight_io_gb: float
    prefetch: bool


def predict(inp: CostModelInput, *, prefetch: bool) -> CostModelPrediction:
    n_microbatches = math.ceil(inp.n_samples / inp.microbatch_size)
    compute_s = inp.fwd_per_microbatch_s * n_microbatches
    load_s = inp.weight_load_s

    if prefetch:
        per_layer_s = max(load_s, compute_s)
    else:
        per_layer_s = load_s + compute_s

    total_wall_s = inp.embed_s + per_layer_s * inp.n_layers
    total_tokens = inp.n_samples * inp.seq_len
    throughput = total_tokens / total_wall_s if total_wall_s > 0 else 0.0

    weight_io_gb = inp.layer_weight_bytes * inp.n_layers / 1e9

    if per_layer_s > 0:
        load_pct = load_s / per_layer_s
        compute_pct = compute_s / per_layer_s
    else:
        load_pct = 0.0
        compute_pct = 0.0

    ratio = load_s / compute_s if compute_s > 0 else float("inf")
    if ratio > 1.2:
        bottleneck = "load"
    elif ratio < 1 / 1.2:
        bottleneck = "compute"
    else:
        bottleneck = "balanced"

    return CostModelPrediction(
        per_layer_s=per_layer_s,
        total_wall_s=total_wall_s,
        throughput_tok_s=throughput,
        bottleneck=bottleneck,
        load_pct=load_pct,
        compute_pct=compute_pct,
        weight_io_gb=weight_io_gb,
        prefetch=prefetch,
    )


@dataclass(frozen=True)
class Recommendation:
    input: CostModelInput
    prefetch: bool
    prediction: CostModelPrediction


def recommend(
    candidates: Sequence[tuple[CostModelInput, bool]],
) -> Recommendation:
    if not candidates:
        raise ValueError("candidates must be non-empty")

    best: Recommendation | None = None
    for inp, pf in candidates:
        pred = predict(inp, prefetch=pf)
        if best is None or pred.throughput_tok_s > best.prediction.throughput_tok_s:
            best = Recommendation(input=inp, prefetch=pf, prediction=pred)

    assert best is not None
    return best
