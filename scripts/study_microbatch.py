"""Microbatch size sweep: measure actual throughput at each power-of-2 mb size.

Collects per-layer profile data (load, fwd, write, VRAM) for each microbatch
candidate. Outputs a CSV for analysis.

Usage:
    uv run scripts/study_microbatch.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --n-samples 1024 --seq-len 128

    uv run scripts/study_microbatch.py \
        --model meta-llama/Llama-3.3-70B-Instruct \
        --n-samples 1024 --seq-len 128
"""
from __future__ import annotations

import argparse
import csv
import time

import torch
from transformers import AutoTokenizer

from fpwap import Callback, Sweep
from fpwap.engine import ProgressEvent
from fpwap.loader import resolve_snapshot_dir
from fpwap.types import HookName


class NullCapture(Callback):
    phase = "read"
    target_layers = "all"
    target_hooks: tuple[HookName, ...] = ("residual_post",)

    def on_batch(self, layer_idx, hook, acts, sample_ids):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", default=None, help="CSV output path")
    parser.add_argument(
        "--mb-candidates",
        default=None,
        help="comma-separated microbatch sizes to test (default: powers of 2)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    snapshot = resolve_snapshot_dir(args.model)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompts = [
        f"The quick brown fox jumps over the lazy dog number {i}."
        for i in range(args.n_samples)
    ]
    batch = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=args.seq_len,
        return_tensors="pt",
    )
    dataset = [
        {
            "input_ids": batch["input_ids"][i : i + 1].to(device),
            "attention_mask": batch["attention_mask"][i : i + 1].to(device),
        }
        for i in range(args.n_samples)
    ]

    total_tokens = args.n_samples * args.seq_len

    if args.mb_candidates:
        candidates = [int(x) for x in args.mb_candidates.split(",")]
    else:
        candidates = []
        mb = 1
        while mb <= args.n_samples:
            candidates.append(mb)
            mb *= 2

    log = args.output or f"/tmp/mb_sweep_{args.model.split('/')[-1]}.csv"
    print(f"model: {args.model}", flush=True)
    print(f"n_samples={args.n_samples} seq_len={args.seq_len}", flush=True)
    print(f"candidates: {candidates}", flush=True)
    print(f"output: {log}", flush=True)
    print(flush=True)

    rows: list[dict] = []

    for mb in candidates:
        if mb > args.n_samples:
            continue

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        n_microbatches = -(-args.n_samples // mb)

        def progress(event: ProgressEvent) -> None:
            pass

        sweep = Sweep(
            model=str(snapshot),
            dataset=dataset,
            seq_len=args.seq_len,
            callbacks=[NullCapture()],
            transport_dtype=dtype,
            seed=0,
            microbatch_size=mb,
            buffer_device="cpu",
            progress=progress,
            execution_device=str(device),
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = sweep.run()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        tps = total_tokens / wall

        total_load = sum(t.load_s for t in result.profile.per_layer.values())
        total_fwd = sum(t.forward_s for t in result.profile.per_layer.values())
        total_cb = sum(t.callback_s for t in result.profile.per_layer.values())
        total_write = sum(t.write_s for t in result.profile.per_layer.values())
        peak_vram = result.profile.peak_vram_gb()
        n_layers = len(result.profile.per_layer)
        per_layer_s = wall / n_layers if n_layers > 0 else 0
        per_sample_us = wall / args.n_samples * 1e6

        row = {
            "model": args.model,
            "n_samples": args.n_samples,
            "seq_len": args.seq_len,
            "microbatch": mb,
            "n_microbatches": n_microbatches,
            "wall_s": round(wall, 2),
            "tok_s": round(tps, 1),
            "per_layer_s": round(per_layer_s, 4),
            "per_sample_us": round(per_sample_us, 1),
            "total_load_s": round(total_load, 2),
            "total_fwd_s": round(total_fwd, 2),
            "total_cb_s": round(total_cb, 3),
            "total_write_s": round(total_write, 3),
            "peak_vram_gb": round(peak_vram, 2),
            "n_layers": n_layers,
        }
        rows.append(row)

        print(
            f"mb={mb:>5d}  n_mb={n_microbatches:>4d}  "
            f"wall={wall:>7.2f}s  tok/s={tps:>9,.1f}  "
            f"per_layer={per_layer_s:.4f}s  "
            f"VRAM={peak_vram:.1f}GB  "
            f"load={total_load:.2f}s  fwd={total_fwd:.2f}s",
            flush=True,
        )

    if rows:
        with open(log, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {log}", flush=True)


if __name__ == "__main__":
    main()
