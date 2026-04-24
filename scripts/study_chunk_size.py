"""Chunk-size sweep: measure throughput at each candidate layers-per-chunk.

Collects wall-clock, tok/s, per-layer breakdown, and peak VRAM for each
chunk_size candidate. Outputs a CSV for analysis.

Usage:
    uv run scripts/study_chunk_size.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --n-samples 1024 --seq-len 128

    uv run scripts/study_chunk_size.py \
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
    parser.add_argument("--microbatch-size", type=int, default=64)
    parser.add_argument("--output", default=None, help="CSV output path")
    parser.add_argument(
        "--candidates",
        default=None,
        help="comma-separated chunk sizes to test (default: 1,2,4,8,16)",
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

    if args.candidates:
        candidates = [int(x) for x in args.candidates.split(",")]
    else:
        candidates = [1, 2, 4, 8, 16]

    log = args.output or f"/tmp/chunk_sweep_{args.model.split('/')[-1]}.csv"
    print(f"model: {args.model}", flush=True)
    print(
        f"n_samples={args.n_samples} seq_len={args.seq_len} "
        f"mb={args.microbatch_size}",
        flush=True,
    )
    print(f"candidates: {candidates}", flush=True)
    print(f"output: {log}", flush=True)
    print(flush=True)

    rows: list[dict] = []

    for cs in candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        layers_done = 0

        def progress(event: ProgressEvent) -> None:
            nonlocal layers_done
            if event.kind == "layer_end":
                layers_done = event.layer_idx + 1
                elapsed = event.wall_s
                tps = total_tokens * layers_done / (elapsed * 32) if elapsed > 0 else 0
                print(
                    f"  layer {event.layer_idx}  {elapsed:.1f}s",
                    flush=True,
                )

        sweep = Sweep(
            model=str(snapshot),
            dataset=dataset,
            seq_len=args.seq_len,
            callbacks=[NullCapture()],
            transport_dtype=dtype,
            seed=0,
            microbatch_size=args.microbatch_size,
            buffer_device="cpu",
            progress=progress,
            execution_device=str(device),
            chunk_size=cs,
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = sweep.run()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        tps = total_tokens / wall

        total_load = sum(t.load_s for t in result.profile.per_layer.values())
        total_fwd = sum(
            t.forward_s for t in result.profile.per_layer.values()
        )
        total_cb = sum(
            t.callback_s for t in result.profile.per_layer.values()
        )
        total_write = sum(
            t.write_s for t in result.profile.per_layer.values()
        )
        total_buf_bytes = sum(
            t.bytes_buffer for t in result.profile.per_layer.values()
        )
        n_layers = len(result.profile.per_layer)

        row = {
            "model": args.model,
            "n_samples": args.n_samples,
            "seq_len": args.seq_len,
            "microbatch": args.microbatch_size,
            "chunk_size": cs,
            "wall_s": round(wall, 2),
            "tok_s": round(tps, 1),
            "total_load_s": round(total_load, 2),
            "total_fwd_s": round(total_fwd, 2),
            "total_cb_s": round(total_cb, 3),
            "total_write_s": round(total_write, 3),
            "total_buf_MB": round(total_buf_bytes / 1e6, 1),
            "n_layers": n_layers,
        }
        rows.append(row)

        print(
            f"chunk_size={cs:>3d}  wall={wall:>7.2f}s  tok/s={tps:>9,.1f}  "
            f"load={total_load:.2f}s  fwd={total_fwd:.2f}s  "
            f"buf_MB={total_buf_bytes / 1e6:.0f}",
            flush=True,
        )
        print(flush=True)

    if rows:
        with open(log, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {log}", flush=True)


if __name__ == "__main__":
    main()
