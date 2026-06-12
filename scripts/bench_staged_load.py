"""A/B benchmark: pinned-staged layer loads vs legacy per-tensor loads.

Reproduces the lens m3 backfill symptom (load-bound MoE sweeps, flat
~53 s regardless of token count) and measures the staged-load fix on the
same workload shape: one unit, microbatch 1, sweep truncated at layer 41.

Usage:
    uv run scripts/bench_staged_load.py \\
        --snapshot ~/models/qwen3-coder-30b-a3b-instruct-fp8-dequant \\
        --seq-len 16384

Runs staged → legacy → staged (interleaved to control page-cache warmth),
prints per-layer load/forward breakdowns, and checks the captured
activations are bitwise identical across paths.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

from fpwap import Sweep
from fpwap.callbacks.common import RawActivations


def run_once(snapshot: str, seq_len: int, staged: bool) -> tuple[float, object]:
    os.environ["FPWAP_STAGED_LOAD"] = "1" if staged else "0"
    torch.manual_seed(0)
    item = {
        "input_ids": torch.randint(0, 50_000, (seq_len,), dtype=torch.long),
        "attention_mask": torch.ones(seq_len, dtype=torch.long),
    }
    sweep = Sweep(
        model=snapshot,
        dataset=[item],
        seq_len=seq_len,
        callbacks=[
            RawActivations(layers=[24, 41], hook="residual_post", last_token_only=False)
        ],
        microbatch_size=1,
        execution_device="cuda",
        progress=False,
    )
    t0 = time.perf_counter()
    result = sweep.run()
    wall = time.perf_counter() - t0
    return wall, result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--seq-len", type=int, default=16384)
    args = ap.parse_args()
    snapshot = str(Path(args.snapshot).expanduser())

    runs: list[tuple[str, float, object]] = []
    for label, staged in (("staged#1", True), ("legacy", False), ("staged#2", True)):
        wall, result = run_once(snapshot, args.seq_len, staged)
        prof = result.profile
        load = sum(t.load_s for t in prof.per_layer.values())
        fwd = sum(t.forward_s for t in prof.per_layer.values())
        n_layers = len(prof.per_layer)
        print(
            f"{label:9s} wall {wall:7.2f}s | {n_layers} layers | "
            f"load {load:6.2f}s  fwd {fwd:6.2f}s | "
            f"{wall / n_layers:5.2f} s/layer"
        )
        runs.append((label, wall, result))

    ref = runs[1][2]
    for label, _, result in (runs[0], runs[2]):
        for layer in (24, 41):
            a = ref.activations(layer, "residual_post")
            b = result.activations(layer, "residual_post")
            exact = torch.equal(a, b)
            print(f"bitwise {label} vs legacy @ layer {layer}: {exact}")
            if not exact:
                diff = (a.float() - b.float()).abs().max().item()
                print(f"  max abs diff: {diff}")


if __name__ == "__main__":
    main()
