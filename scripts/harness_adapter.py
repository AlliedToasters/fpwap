"""fpwap adapter for lmprobe benchmark_offload_backends.py harness.

Drop-in ``run_fpwap`` function matching the harness contract::

    (load_s, extract_times, fingerprint, peak_VRAM)

Uses the Extractor API so multiple iters reuse the cached empty model
and accel_index — no per-iteration rebuild.

Standalone smoke test::

    uv run scripts/harness_adapter.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --n-samples 32 --seq-len 128 --iters 3
"""
from __future__ import annotations

import argparse
import time

import torch
from torch import Tensor

from fpwap import Callback, Extractor
from fpwap.types import BatchResult, HookName


class _FingerprintCapture(Callback):
    """Captures residual_post at specified layers for fingerprinting."""

    phase = "read"

    def __init__(
        self,
        n_samples: int,
        seq_len: int,
        hidden_size: int,
        layers: list[int],
        dtype: torch.dtype,
    ) -> None:
        self.target_layers = layers
        self.target_hooks: tuple[HookName, ...] = ("residual_post",)
        self.acts: dict[int, Tensor] = {
            i: torch.zeros(n_samples, seq_len, hidden_size, dtype=dtype)
            for i in layers
        }

    def on_batch(
        self,
        layer_idx: int,
        hook: HookName,
        acts: Tensor,
        sample_ids: Tensor,
    ) -> BatchResult:
        self.acts[layer_idx][sample_ids] = acts.detach().to(self.acts[layer_idx].dtype).cpu()
        return None


def compute_fingerprint(
    acts: dict[int, Tensor],
    attention_mask: Tensor | None,
) -> float:
    """Per-batch mean of non-pad residuals across captured layers.

    Mask pad positions, mean over (sample, position, hidden) dims for each
    layer, then mean across layers -> single scalar determinism check.
    """
    layer_means: list[float] = []
    for _layer_idx, hidden in sorted(acts.items()):
        if attention_mask is not None:
            mask = attention_mask.bool().unsqueeze(-1)
            vals = hidden.float().masked_select(mask)
        else:
            vals = hidden.float().flatten()
        layer_means.append(vals.mean().item())
    return sum(layer_means) / len(layer_means) if layer_means else 0.0


def run_fpwap(
    model_id: str,
    pretokenized_batch: Tensor,
    hf_layers: list[int],
    *,
    iters: int = 1,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    microbatch_size: int | None = None,
    attention_mask: Tensor | None = None,
) -> tuple[float, list[float], float, float]:
    """Run fpwap extraction and return the harness contract tuple.

    Args:
        model_id: HF model ID or local snapshot path.
        pretokenized_batch: [N, S] input_ids tensor.
        hf_layers: Layer indices to extract residual_post from.
        iters: Number of extraction iterations to time.
        device: Execution device for the forward pass.
        dtype: Weight and transport dtype.
        microbatch_size: Samples per microbatch (None = full batch).
        attention_mask: [N, S] attention mask; None = all tokens active.

    Returns:
        load_s: Extractor build time (empty model + accel_index).
        extract_times: Per-iteration wall-clock seconds.
        fingerprint: Per-batch mean of non-pad residuals (scalar).
        peak_VRAM: Peak GPU memory in bytes via torch.cuda.max_memory_allocated.
    """
    use_cuda = torch.cuda.is_available() and "cuda" in device

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    ext = Extractor.from_hf(model_id, dtype=dtype)
    if use_cuda:
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    config = getattr(ext._model, "config", None)
    hidden_size = int(getattr(config, "hidden_size", 0))
    if hidden_size == 0:
        raise ValueError(f"could not determine hidden_size from {model_id}")

    n_samples, seq_len = pretokenized_batch.shape

    dataset: list[dict[str, Tensor]] = []
    for i in range(n_samples):
        item: dict[str, Tensor] = {"input_ids": pretokenized_batch[i : i + 1]}
        if attention_mask is not None:
            item["attention_mask"] = attention_mask[i : i + 1]
        dataset.append(item)

    extract_times: list[float] = []
    fingerprint = 0.0

    for _ in range(iters):
        cap = _FingerprintCapture(
            n_samples=n_samples,
            seq_len=seq_len,
            hidden_size=hidden_size,
            layers=hf_layers,
            dtype=dtype,
        )
        sweep = ext.sweep(
            dataset=dataset,
            seq_len=seq_len,
            callbacks=[cap],
            transport_dtype=dtype,
            execution_device=device,
            microbatch_size=microbatch_size,
            seed=0,
            progress=False,
            apply_final_norm=False,
        )

        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        sweep.run()
        if use_cuda:
            torch.cuda.synchronize()
        extract_times.append(time.perf_counter() - t0)

        fingerprint = compute_fingerprint(cap.acts, attention_mask)

    peak_vram = float(torch.cuda.max_memory_allocated()) if use_cuda else 0.0

    return load_s, extract_times, fingerprint, peak_vram


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model ID or local snapshot path")
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--microbatch", type=int, default=None)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from fpwap.loader import resolve_snapshot_dir

    dtype = getattr(torch, args.dtype)
    snapshot_dir = resolve_snapshot_dir(args.model)
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
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

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(snapshot_dir)
    n_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None)
    if n_layers is None:
        raise ValueError(f"could not determine n_layers from {args.model}")
    hf_layers_count: int = int(n_layers)
    hf_layers = list(range(hf_layers_count))

    load_s, extract_times, fingerprint, peak_vram = run_fpwap(
        model_id=args.model,
        pretokenized_batch=batch["input_ids"],
        hf_layers=hf_layers,
        iters=args.iters,
        device=args.device,
        dtype=dtype,
        microbatch_size=args.microbatch,
        attention_mask=batch["attention_mask"],
    )

    print(f"=== harness_adapter: {args.model} ===")
    print(f"  load_s          : {load_s:.3f}")
    print(f"  extract_times   : {[f'{t:.3f}' for t in extract_times]}")
    print(f"  fingerprint     : {fingerprint:.8f}")
    print(f"  peak_VRAM (MB)  : {peak_vram / 1e6:.1f}")
    print(f"  iters           : {args.iters}")
    print(f"  N               : {args.n_samples}")
    print(f"  seq_len         : {args.seq_len}")
    print(f"  layers          : {len(hf_layers)}")
    median_t = sorted(extract_times)[len(extract_times) // 2]
    print(f"  median extract  : {median_t:.3f} s")
    total_tokens = args.n_samples * args.seq_len
    print(f"  median tok/s    : {total_tokens / median_t:,.1f}")


if __name__ == "__main__":
    main()
