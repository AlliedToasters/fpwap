"""Benchmark: fixed vs bucketed padding on mixed-length datasets.

Demonstrates the compute savings from padding='bucketed' on a bimodal
sequence-length distribution (the "alpaca ~50 + ai_liar 400+" use case
from issue #10).

Usage:
    uv run scripts/bench_bucketed_padding.py
    uv run scripts/bench_bucketed_padding.py --n-short 100 --n-long 100
    uv run scripts/bench_bucketed_padding.py --n-layers 6 --hidden 128
"""
from __future__ import annotations

import argparse
import time
import warnings

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from fpwap import Callback, Sweep
from fpwap.types import BatchResult, HookName


class NullCapture(Callback):
    phase = "read"
    target_layers = "all"
    target_hooks: tuple[HookName, ...] = ("residual_post",)

    def __init__(self) -> None:
        self.touches = 0

    def on_batch(
        self, layer_idx: int, hook: HookName,
        acts: torch.Tensor, sample_ids: torch.Tensor,
    ) -> BatchResult:
        self.touches += int(acts.shape[0])
        return None


def build_mixed_dataset(
    n_short: int, n_long: int, short_range: tuple[int, int],
    long_range: tuple[int, int], max_seq: int, vocab: int,
) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(42)
    items: list[dict[str, torch.Tensor]] = []
    for low, high, count in [
        (short_range[0], short_range[1], n_short),
        (long_range[0], long_range[1], n_long),
    ]:
        for _ in range(count):
            L = torch.randint(low, high + 1, (1,)).item()
            ids = torch.full((1, max_seq), 0, dtype=torch.long)
            mask = torch.zeros((1, max_seq), dtype=torch.long)
            ids[0, max_seq - L:] = torch.randint(1, vocab, (L,))
            mask[0, max_seq - L:] = 1
            items.append({"input_ids": ids, "attention_mask": mask})
    return items


def run_sweep(model, items, seq_len, padding, label):
    cap = NullCapture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sweep = Sweep(
            model=model,
            dataset=items,
            seq_len=seq_len,
            callbacks=[cap],
            transport_dtype=torch.float32,
            padding=padding,
            apply_final_norm=False,
            progress=False,
        )
        t0 = time.perf_counter()
        result = sweep.run()
        wall = time.perf_counter() - t0

    prof = result.profile
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  wall clock:   {wall:.4f}s")
    print(f"  tokens:       {prof.total_tokens:,}")
    print(f"  throughput:   {prof.throughput_tok_per_s():,.0f} tok/s")
    print(f"  embed:        {prof.embed_s:.4f}s")
    print(f"  loop:         {prof.loop_s:.4f}s")
    print(f"  cb touches:   {cap.touches}")
    return wall, prof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-short", type=int, default=50)
    p.add_argument("--n-long", type=int, default=50)
    p.add_argument("--short-min", type=int, default=30)
    p.add_argument("--short-max", type=int, default=70)
    p.add_argument("--long-min", type=int, default=350)
    p.add_argument("--long-max", type=int, default=450)
    p.add_argument("--max-seq", type=int, default=512)
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    args = p.parse_args()

    config = GPT2Config(
        vocab_size=args.vocab,
        n_positions=args.max_seq,
        n_embd=args.hidden,
        n_layer=args.n_layers,
        n_head=args.n_heads,
    )
    torch.manual_seed(0)
    model = GPT2LMHeadModel(config)
    model.eval()

    items = build_mixed_dataset(
        args.n_short, args.n_long,
        (args.short_min, args.short_max),
        (args.long_min, args.long_max),
        args.max_seq, args.vocab,
    )
    n_total = len(items)
    real_lengths = [it["attention_mask"].sum().item() for it in items]
    avg_len = sum(real_lengths) / len(real_lengths)
    print(f"\nDataset: {n_total} items, max_seq={args.max_seq}")
    print(f"  short: {args.n_short} items, lengths {args.short_min}-{args.short_max}")
    print(f"  long:  {args.n_long} items, lengths {args.long_min}-{args.long_max}")
    print(f"  avg real length: {avg_len:.1f} / {args.max_seq} "
          f"({avg_len/args.max_seq*100:.0f}% utilization)")
    print(f"  wasted tokens (fixed): {n_total * args.max_seq - sum(real_lengths):,}")

    print(f"\nModel: GPT-2 config, {args.n_layers}L × {args.hidden}H × {args.n_heads}heads")

    wall_fixed, prof_fixed = run_sweep(
        model, items, args.max_seq, "fixed", "padding='fixed' (pad-to-max)"
    )
    wall_bucketed, prof_bucketed = run_sweep(
        model, items, args.max_seq, "bucketed", "padding='bucketed'"
    )

    speedup = wall_fixed / wall_bucketed if wall_bucketed > 0 else float("inf")
    token_reduction = 1.0 - prof_bucketed.total_tokens / prof_fixed.total_tokens
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  speedup:          {speedup:.2f}×")
    print(f"  token reduction:  {token_reduction*100:.1f}%")
    print(f"  fixed wall:       {wall_fixed:.4f}s")
    print(f"  bucketed wall:    {wall_bucketed:.4f}s")
    print(f"  fixed tokens:     {prof_fixed.total_tokens:,}")
    print(f"  bucketed tokens:  {prof_bucketed.total_tokens:,}")


if __name__ == "__main__":
    main()
