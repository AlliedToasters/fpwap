"""Residual buffer can live on a different device than execution.

Needed for oversized runs where (N × seq × hidden) exceeds VRAM but fits
in host RAM. Buffer on CPU, forward on GPU — must still match naive bit-
exactly at real positions.

CPU-only here (execution_device == buffer_device-or-other); the GPU
streaming integration of this path is exercised by the benchmark script.
"""
from __future__ import annotations

import pytest
import torch

SEED = 0
N_SAMPLES = 6
SEQ_LEN = 8
HIDDEN = 32
N_LAYERS = 2
N_HEAD = 2
VOCAB = 100


def _tiny_gpt2() -> torch.nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=VOCAB,
        n_positions=SEQ_LEN,
        n_embd=HIDDEN,
        n_layer=N_LAYERS,
        n_head=N_HEAD,
    )
    torch.manual_seed(SEED)
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


@pytest.mark.integration
def test_buffer_device_split_matches_naive() -> None:
    """Execution on CPU, buffer explicitly set to CPU (smoke test for the path)."""
    from fpwap import Callback, Sweep
    from fpwap.types import BatchResult, HookName

    model = _tiny_gpt2()
    torch.manual_seed(SEED + 1)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    baseline: dict[int, torch.Tensor] = {}

    def _make_hook(layer_idx: int):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            baseline[layer_idx] = h.detach().clone()

        return hook

    handles = []
    for i, block in enumerate(model.transformer.h):
        handles.append(block.register_forward_hook(_make_hook(i)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()

    class Capture(Callback):
        phase = "read"
        target_layers = "all"
        target_hooks: tuple[HookName, ...] = ("residual_post",)

        def __init__(self) -> None:
            self.acts: dict[int, torch.Tensor] = {
                i: torch.zeros(N_SAMPLES, SEQ_LEN, HIDDEN) for i in range(N_LAYERS)
            }

        def on_batch(
            self,
            layer_idx: int,
            hook: HookName,
            acts: torch.Tensor,
            sample_ids: torch.Tensor,
        ) -> BatchResult:
            self.acts[layer_idx][sample_ids.cpu()] = acts.detach().float().cpu()
            return None

    cap = Capture()
    run = Sweep(
        model=model,
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[cap],
        transport_dtype=torch.float32,
        microbatch_size=2,
        buffer_device="cpu",  # explicit even though exec is also CPU
        seed=SEED,
        apply_final_norm=False,
    )
    run.run()

    for layer_idx, baseline_hidden in baseline.items():
        got = cap.acts[layer_idx]
        max_diff = (got - baseline_hidden).abs().max().item()
        assert torch.allclose(got, baseline_hidden, atol=1e-6, rtol=1e-6), (
            f"split-buffer mismatch at layer {layer_idx}: max abs diff {max_diff}"
        )
