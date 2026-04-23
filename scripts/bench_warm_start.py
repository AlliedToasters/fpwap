"""fpwap multi-dataset warm-start benchmark.

Proves the Extractor handle eliminates per-sweep build_empty_model_and_index
cost across back-to-back sweeps. The multi-eval-dataset pattern (15+ datasets
per experiment pass) is the motivating use case.

Two modes:
  warm  — Extractor.from_hf() once, then ext.sweep() × N (handle reuse)
  cold  — Sweep(model=str) × N (rebuilds empty model + accel_index each time)

Reports: total wall-clock, setup portion, per-sweep median, speedup ratio.

Usage:
    uv run scripts/bench_warm_start.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --n-sweeps 15 --n-samples 32 --seq-len 128

    uv run scripts/bench_warm_start.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --n-sweeps 15 --mode cold \\
        --n-samples 32 --seq-len 128
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from fpwap import Callback, Extractor, Sweep
from fpwap.loader import resolve_snapshot_dir
from fpwap.types import BatchResult, HookName


class NullCapture(Callback):
    """Minimal callback — forces forward work without shaping measurement."""

    phase = "read"
    target_layers = "all"
    target_hooks: tuple[HookName, ...] = ("residual_post",)

    def __init__(self) -> None:
        self.touches = 0

    def on_batch(
        self,
        layer_idx: int,
        hook: HookName,
        acts: torch.Tensor,
        sample_ids: torch.Tensor,
    ) -> BatchResult:
        self.touches += int(acts.shape[0])
        return None


def summarize_timings(
    setup_s: float,
    sweep_times: list[float],
) -> dict[str, float]:
    """Compute summary statistics from raw warm-start timing data.

    Returns a dict with: n_sweeps, setup_s, total_run_s, total_s,
    median_sweep_s, mean_sweep_s, min_sweep_s, max_sweep_s,
    setup_fraction.
    """
    total_run_s = sum(sweep_times)
    total_s = setup_s + total_run_s
    n = len(sweep_times)
    return {
        "n_sweeps": float(n),
        "setup_s": setup_s,
        "total_run_s": total_run_s,
        "total_s": total_s,
        "median_sweep_s": statistics.median(sweep_times) if n > 0 else 0.0,
        "mean_sweep_s": statistics.mean(sweep_times) if n > 0 else 0.0,
        "min_sweep_s": min(sweep_times) if n > 0 else 0.0,
        "max_sweep_s": max(sweep_times) if n > 0 else 0.0,
        "setup_fraction": setup_s / total_s if total_s > 0 else 0.0,
    }


def _build_dataset(
    tokenizer: Any,
    n_samples: int,
    seq_len: int,
) -> list[dict[str, torch.Tensor]]:
    prompts = [
        f"The quick brown fox jumps over the lazy dog number {i}."
        for i in range(n_samples)
    ]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    batch = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    input_ids: torch.Tensor = batch["input_ids"]
    attention_mask: torch.Tensor = batch["attention_mask"]
    return [
        {
            "input_ids": input_ids[i : i + 1],
            "attention_mask": attention_mask[i : i + 1],
        }
        for i in range(n_samples)
    ]


def _run_warm(
    model_id: str,
    dataset: list[dict[str, torch.Tensor]],
    n_sweeps: int,
    seq_len: int,
    dtype: torch.dtype,
    device: str,
    microbatch: int | None,
    buffer_device: str | None,
) -> tuple[float, list[float]]:
    """Warm path: Extractor.from_hf() once, then N sweeps reusing handle."""
    use_cuda = torch.cuda.is_available() and "cuda" in device

    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    ext = Extractor.from_hf(model_id, dtype=dtype)
    if use_cuda:
        torch.cuda.synchronize()
    setup_s = time.perf_counter() - t0

    sweep_times: list[float] = []
    for _ in range(n_sweeps):
        cb = NullCapture()
        sweep = ext.sweep(
            dataset=dataset,
            seq_len=seq_len,
            callbacks=[cb],
            transport_dtype=dtype,
            execution_device=device,
            microbatch_size=microbatch,
            seed=0,
            progress=False,
            apply_final_norm=False,
            buffer_device=buffer_device,
        )
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        sweep.run()
        if use_cuda:
            torch.cuda.synchronize()
        sweep_times.append(time.perf_counter() - t0)

    return setup_s, sweep_times


def _run_cold(
    model_id: str,
    snapshot_dir: Path,
    dataset: list[dict[str, torch.Tensor]],
    n_sweeps: int,
    seq_len: int,
    dtype: torch.dtype,
    device: str,
    microbatch: int | None,
    buffer_device: str | None,
) -> tuple[float, list[float]]:
    """Cold path: N × Sweep(model=str), each rebuilds empty model + index."""
    use_cuda = torch.cuda.is_available() and "cuda" in device

    sweep_times: list[float] = []
    for _ in range(n_sweeps):
        cb = NullCapture()
        sweep = Sweep(
            model=str(snapshot_dir),
            dataset=dataset,
            seq_len=seq_len,
            callbacks=[cb],
            transport_dtype=dtype,
            execution_device=device,
            microbatch_size=microbatch,
            seed=0,
            progress=False,
            apply_final_norm=False,
            buffer_device=buffer_device,
        )
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        sweep.run()
        if use_cuda:
            torch.cuda.synchronize()
        sweep_times.append(time.perf_counter() - t0)

    return 0.0, sweep_times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model ID or local snapshot path")
    parser.add_argument("--n-sweeps", type=int, default=15)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--microbatch", type=int, default=None)
    parser.add_argument("--mode", choices=["warm", "cold", "both"], default="both")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--buffer-device", default=None)
    parser.add_argument("--out", default=None, help="CSV output path")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    snapshot_dir = resolve_snapshot_dir(args.model)
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    dataset = _build_dataset(tokenizer, args.n_samples, args.seq_len)

    rows: list[dict[str, object]] = []
    header = [
        "model", "mode", "n_sweeps", "n_samples", "seq_len",
        "setup_s", "total_run_s", "total_s",
        "median_sweep_s", "mean_sweep_s", "min_sweep_s", "max_sweep_s",
        "setup_fraction",
    ]

    modes = ["warm", "cold"] if args.mode == "both" else [args.mode]
    for mode in modes:
        print(f"\n[bench_warm_start] {mode} — {args.model}, {args.n_sweeps} sweeps")
        if mode == "warm":
            setup_s, sweep_times = _run_warm(
                model_id=args.model,
                dataset=dataset,
                n_sweeps=args.n_sweeps,
                seq_len=args.seq_len,
                dtype=dtype,
                device=args.device,
                microbatch=args.microbatch,
                buffer_device=args.buffer_device,
            )
        else:
            setup_s, sweep_times = _run_cold(
                model_id=args.model,
                snapshot_dir=snapshot_dir,
                dataset=dataset,
                n_sweeps=args.n_sweeps,
                seq_len=args.seq_len,
                dtype=dtype,
                device=args.device,
                microbatch=args.microbatch,
                buffer_device=args.buffer_device,
            )

        stats = summarize_timings(setup_s, sweep_times)
        row: dict[str, object] = {
            "model": args.model,
            "mode": mode,
            "n_sweeps": args.n_sweeps,
            "n_samples": args.n_samples,
            "seq_len": args.seq_len,
        }
        for k, v in stats.items():
            if k == "n_sweeps":
                continue
            row[k] = f"{v:.4f}"
        rows.append(row)

        print(f"  setup            : {stats['setup_s']:.3f} s")
        print(f"  total run        : {stats['total_run_s']:.3f} s")
        print(f"  total            : {stats['total_s']:.3f} s")
        print(f"  median sweep     : {stats['median_sweep_s']:.4f} s")
        print(f"  mean sweep       : {stats['mean_sweep_s']:.4f} s")
        print(f"  min/max sweep    : {stats['min_sweep_s']:.4f} / {stats['max_sweep_s']:.4f} s")
        print(f"  setup fraction   : {stats['setup_fraction']:.1%}")
        print(f"  per-sweep times  : {[f'{t:.3f}' for t in sweep_times]}")

    if len(rows) == 2:
        warm_total = float(rows[0]["total_s"])  # type: ignore[arg-type]
        cold_total = float(rows[1]["total_s"])  # type: ignore[arg-type]
        if warm_total > 0:
            print(f"\n  speedup (cold/warm): {cold_total / warm_total:.2f}×")

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
