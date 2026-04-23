"""fpwap N-scaling benchmark — the amortization curve.

fpwap's thesis is weight-I/O O(N_layers) instead of O(N_batches × N_layers).
On a single-batch timer fpwap can only tie per-forward streaming (accelerate
with model resident on CPU/disk); the win appears at dataset scale where
fpwap amortizes 80 layer-loads across the full N while naive pays them
per microbatch.

This script sweeps N for a fixed model/seq_len and dumps (wall_s,
throughput, peak_gpu_mb, peak_cpu_mb) per N. Output is CSV so it merges
with the team's existing benchmark harness without format surgery.

Usage (hero curve — the killer plot):
    uv run scripts/bench_n_scaling.py \\
        --model meta-llama/Llama-3.3-70B-Instruct \\
        --seq-len 256 --microbatch 128 --mode streaming \\
        --ns 8,64,512,4096 --buffer-device cpu \\
        --out /tmp/fpwap_70b_n_scaling.csv
"""
from __future__ import annotations

import argparse
import csv
import resource
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fpwap import Callback, Sweep
from fpwap.loader import resolve_snapshot_dir
from fpwap.types import BatchResult, HookName


class NullCapture(Callback):
    """Minimal callback — forces the forward work to land without shaping
    the measurement. One int-reduction per microbatch per layer."""

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


def _build_dataset(
    tokenizer,
    n_samples: int,
    seq_len: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    prompts = [f"The quick brown fox jumps over the lazy dog number {i}." for i in range(n_samples)]
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
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    return [
        {
            "input_ids": input_ids[i : i + 1],
            "attention_mask": attention_mask[i : i + 1],
        }
        for i in range(n_samples)
    ]


def _run_one_n_fpwap(
    snapshot_dir: Path,
    mode: str,
    dataset: list[dict[str, torch.Tensor]],
    dtype: torch.dtype,
    device: torch.device,
    seq_len: int,
    microbatch: int,
    buffer_device: str | None,
) -> tuple[float, int]:
    """One fpwap run at fixed N. Returns (wall_s, n_layers)."""
    cb = NullCapture()
    if mode == "preloaded":
        model_obj = AutoModelForCausalLM.from_pretrained(
            snapshot_dir, dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
        model_obj.eval()
        sweep = Sweep(
            model=model_obj,
            dataset=dataset,
            seq_len=seq_len,
            callbacks=[cb],
            transport_dtype=dtype,
            microbatch_size=microbatch,
            seed=0,
            progress=False,
            buffer_device=buffer_device,
        )
    else:
        sweep = Sweep(
            model=str(snapshot_dir),
            execution_device=str(device),
            dataset=dataset,
            seq_len=seq_len,
            callbacks=[cb],
            transport_dtype=dtype,
            microbatch_size=microbatch,
            seed=0,
            progress=False,
            buffer_device=buffer_device,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = sweep.run()
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0
    return wall_s, len(result.profile.per_layer)


def _run_one_n_naive(
    snapshot_dir: Path,
    dataset: list[dict[str, torch.Tensor]],
    dtype: torch.dtype,
    device: torch.device,
    microbatch: int,
) -> tuple[float, int]:
    """accelerate.cpu_offload baseline at fixed N."""
    from accelerate import cpu_offload

    model = AutoModelForCausalLM.from_pretrained(
        snapshot_dir, dtype=dtype, low_cpu_mem_usage=True
    )
    model.eval()

    inner = getattr(model, "model", None) or getattr(model, "transformer", None)
    blocks = getattr(inner, "layers", None) or getattr(inner, "h", None)
    if blocks is None:
        raise RuntimeError("could not locate transformer blocks")
    n_layers = len(blocks)

    model = cpu_offload(model, execution_device=device)
    n_samples = len(dataset)
    has_mask = "attention_mask" in dataset[0]

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, n_samples, microbatch):
            stop = min(start + microbatch, n_samples)
            input_ids = torch.cat(
                [dataset[i]["input_ids"] for i in range(start, stop)], dim=0
            ).to(device)
            kwargs: dict[str, torch.Tensor] = {"input_ids": input_ids}
            if has_mask:
                kwargs["attention_mask"] = torch.cat(
                    [dataset[i]["attention_mask"] for i in range(start, stop)],
                    dim=0,
                ).to(device)
            model(**kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0
    return wall_s, n_layers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--microbatch", type=int, default=128)
    parser.add_argument("--mode", choices=["preloaded", "streaming", "naive"], default="streaming")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--buffer-device", default=None)
    parser.add_argument(
        "--ns",
        default="8,64,512,4096",
        help="comma-separated N values to sweep",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="CSV output path; stdout only if omitted",
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    ns = [int(x) for x in args.ns.split(",")]

    snapshot_dir = resolve_snapshot_dir(args.model)
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)

    rows: list[tuple[object, ...]] = []
    header: tuple[str, ...] = (
        "model",
        "mode",
        "N",
        "seq_len",
        "microbatch",
        "wall_s",
        "tok_s",
        "peak_gpu_mb",
        "peak_cpu_mb",
        "n_layers",
    )
    print("  ".join(f"{h:>12}" for h in header))

    for n in ns:
        dataset = _build_dataset(tokenizer, n, args.seq_len, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        if args.mode == "naive":
            mb = min(args.microbatch, n)
            wall_s, n_layers = _run_one_n_naive(snapshot_dir, dataset, dtype, device, mb)
        else:
            wall_s, n_layers = _run_one_n_fpwap(
                snapshot_dir,
                args.mode,
                dataset,
                dtype,
                device,
                args.seq_len,
                min(args.microbatch, n),
                args.buffer_device,
            )

        total_tokens = n * args.seq_len
        tok_s = total_tokens / wall_s
        peak_gpu_mb = (
            torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
        )
        peak_cpu_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

        row: tuple[object, ...] = (
            args.model,
            args.mode,
            n,
            args.seq_len,
            min(args.microbatch, n),
            f"{wall_s:.2f}",
            f"{tok_s:.1f}",
            f"{peak_gpu_mb:.1f}",
            f"{peak_cpu_mb:.1f}",
            n_layers,
        )
        rows.append(row)
        print("  ".join(f"{str(v):>12}" for v in row))

        # free before next N to avoid accumulating VRAM/RAM pressure
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
