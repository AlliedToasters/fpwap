"""fpwap throughput benchmark.

Runs a single-pass `residual_post` sweep over a configurable dataset size
and prints the profile breakdown plus end-to-end tokens/sec. Designed to
validate the SPEC §17 hero-model targets.

Usage:
    uv run scripts/benchmark.py --model meta-llama/Llama-3.2-1B-Instruct \\
        --n-samples 128 --seq-len 128 --mode preloaded

    uv run scripts/benchmark.py --model meta-llama/Llama-3.1-8B-Instruct \\
        --n-samples 512 --seq-len 128 --mode streaming

    uv run scripts/benchmark.py --model meta-llama/Llama-3.1-8B-Instruct \\
        --n-samples 512 --seq-len 128 --mode naive

Modes:
    preloaded  — model fits on GPU; fpwap loop runs without weight streaming
    streaming  — model mmap'd from HF cache; per-layer load/unload each sweep
    naive      — accelerate.cpu_offload baseline; weights streamed per-forward
                 via AlignDevicesHook. The SPEC §17 reference point that fpwap
                 beats by ≥4× at hero scale.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from fpwap import Callback, Sweep
from fpwap.types import BatchResult, HookName


class NullCapture(Callback):
    """Minimal callback: touches each activation so the forward work isn't
    optimized away, and counts layers × microbatches for sanity."""

    phase = "read"
    target_layers = "all"
    target_hooks: tuple[HookName, ...] = ("residual_post",)

    def __init__(self) -> None:
        self.touches = 0
        self.last_token_norms: list[float] = []

    def on_batch(
        self,
        layer_idx: int,
        hook: HookName,
        acts: torch.Tensor,
        sample_ids: torch.Tensor,
    ) -> BatchResult:
        # A single scalar reduction forces the kernel to actually run; very
        # cheap compared to the forward itself.
        self.touches += int(acts.shape[0])
        if layer_idx == 0 and len(self.last_token_norms) < 4:
            norm = acts[:, -1, :].float().norm(dim=-1).mean().item()
            self.last_token_norms.append(norm)
        return None


def _build_synthetic_dataset(
    tokenizer,
    n_samples: int,
    seq_len: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """Synthetic prompts → left-padded (input_ids, attention_mask) items."""
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


def _run_naive(
    snapshot_dir: Path,
    dataset: list[dict[str, torch.Tensor]],
    dtype: torch.dtype,
    device: torch.device,
    microbatch: int,
) -> tuple[float, int, int]:
    """accelerate.cpu_offload baseline (SPEC §17 reference point).

    Loads the model onto CPU, installs AlignDevicesHooks so weights stream
    onto `device` per-forward, then walks the dataset in microbatches.
    Forward hooks on each transformer block force a residual touch so the
    shape of the work matches fpwap's NullCapture (otherwise a layer-only
    baseline would be unfairly cheap — no per-layer activation egress).

    Returns (wall_time_seconds, total_tokens, n_layers).
    """
    from accelerate import cpu_offload

    model = AutoModelForCausalLM.from_pretrained(
        snapshot_dir, dtype=dtype, low_cpu_mem_usage=True
    )
    model.eval()

    # Find transformer blocks structurally — same discipline as fpwap's plumbing.
    inner = getattr(model, "model", None) or getattr(model, "transformer", None)
    blocks = getattr(inner, "layers", None) or getattr(inner, "h", None)
    if blocks is None:
        raise RuntimeError("could not locate transformer blocks on this model")

    touches = [0] * len(blocks)

    def make_hook(i: int):
        def _h(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            touches[i] += int(t.shape[0])

        return _h

    handles = [b.register_forward_hook(make_hook(i)) for i, b in enumerate(blocks)]

    model = cpu_offload(model, execution_device=device)

    n_samples = len(dataset)
    seq_len = int(dataset[0]["input_ids"].shape[-1])
    total_tokens = n_samples * seq_len

    has_mask = "attention_mask" in dataset[0]

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
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
    finally:
        for h in handles:
            h.remove()
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0
    return wall_s, total_tokens, len(blocks)


def _resolve_snapshot_dir(model_id: str) -> Path:
    """Return an on-disk path for the model. Uses HF cache if present; does
    not download. Accepts either an HF hub id or an absolute path."""
    p = Path(model_id)
    if p.is_dir():
        return p
    return Path(snapshot_download(model_id, local_files_only=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id or local snapshot path")
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--microbatch", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["preloaded", "streaming", "naive"],
        default="streaming",
        help=(
            "preloaded = full model on GPU; streaming = per-layer mmap "
            "load/unload; naive = accelerate.cpu_offload (SPEC §17 baseline)"
        ),
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--buffer-device",
        default=None,
        help="residual-buffer device (e.g. cpu for oversized runs); defaults to execution device",
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    snapshot_dir = _resolve_snapshot_dir(args.model)
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    dataset = _build_synthetic_dataset(tokenizer, args.n_samples, args.seq_len, device)

    if args.mode == "naive":
        mb = args.microbatch if args.microbatch is not None else 1
        print(f"[benchmark] naive cpu_offload: {args.model} from {snapshot_dir}")
        wall_s, total_tokens, n_layers = _run_naive(
            snapshot_dir, dataset, dtype, device, mb
        )
        throughput = total_tokens / wall_s
        print()
        print(f"=== benchmark: {args.model} (naive/cpu_offload) ===")
        print(f"  samples         : {args.n_samples}")
        print(f"  seq_len         : {args.seq_len}")
        print(f"  microbatch      : {mb}")
        print(f"  dtype           : {args.dtype}")
        print(f"  total tokens    : {total_tokens:,}")
        print(f"  wall time       : {wall_s:.2f} s")
        print(f"  throughput      : {throughput:,.1f} tok/s")
        print(f"  layers          : {n_layers}")
        print()
        print("Compare against fpwap: fpwap_tok_s / this_tok_s should be >= 4 at hero scale.")
        return

    if args.mode == "preloaded":
        print(f"[benchmark] pre-loading {args.model} onto {device} as {args.dtype}")
        model = AutoModelForCausalLM.from_pretrained(
            snapshot_dir, dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
        model.eval()
        sweep_model: str | torch.nn.Module = model
        execution_device: str | None = None
    else:
        print(f"[benchmark] streaming {args.model} from {snapshot_dir}")
        sweep_model = str(snapshot_dir)
        execution_device = str(device)

    cb = NullCapture()
    kwargs = {
        "model": sweep_model,
        "dataset": dataset,
        "seq_len": args.seq_len,
        "callbacks": [cb],
        "transport_dtype": dtype,
        "seed": 0,
    }
    if args.mode == "streaming":
        kwargs["execution_device"] = execution_device
    if args.microbatch is not None:
        kwargs["microbatch_size"] = args.microbatch
    if args.buffer_device is not None:
        kwargs["buffer_device"] = args.buffer_device

    sweep = Sweep(**kwargs)

    # Warm up allocator and any lazy init; don't count toward wall clock.
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = sweep.run()
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0

    total_tokens = args.n_samples * args.seq_len
    throughput = total_tokens / wall_s

    print()
    print(f"=== benchmark: {args.model} ({args.mode}) ===")
    print(f"  samples         : {args.n_samples}")
    print(f"  seq_len         : {args.seq_len}")
    print(f"  dtype           : {args.dtype}")
    print(f"  total tokens    : {total_tokens:,}")
    print(f"  wall time       : {wall_s:.2f} s")
    print(f"  throughput      : {throughput:,.1f} tok/s  (harness)")
    print(
        f"  throughput (api): {result.profile.throughput_tok_per_s():,.1f} tok/s  "
        f"(excludes warmup)"
    )
    print(
        f"  weight bw       : {result.profile.weight_bandwidth_gb_per_s():.2f} GB/s "
        f"sustained"
    )
    print(f"  layers          : {len(result.profile.per_layer)}")
    print(f"  callback touches: {cb.touches:,}")
    print(f"  weight bytes    : {result.profile.bytes_moved()['weights'] / 1e9:.2f} GB")
    print(f"  buffer bytes    : {result.profile.bytes_moved()['buffer'] / 1e9:.2f} GB")
    slow_layer, slow_phase = result.profile.slowest_layer()
    print(f"  slowest layer   : {slow_layer} phase={slow_phase}")
    print()
    print(result.profile.summary())


if __name__ == "__main__":
    main()
