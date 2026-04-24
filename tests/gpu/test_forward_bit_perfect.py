"""GPU bit-perfect contract: fpwap vs. naive forward on CUDA.

Per SPEC.md §14.2 and CLAUDE.md principle 2, every forward pass must match
bit-for-bit (within bf16 tolerance) against a reference path running on the
same weights, same dtype, same device.

This test runs the core inverted loop on real CUDA hardware to catch GPU-only
bugs the CPU integration tests can miss (bf16 matmul quirks, synchronization,
per-device non-determinism). The streaming / cpu_offload comparison is
separate — here we only care the loop itself is correct on-device.
"""
from __future__ import annotations

import pytest
import torch

SEED = 0
N_SAMPLES = 8
SEQ_LEN = 16
HIDDEN = 64
N_LAYERS = 4
N_HEAD = 4
VOCAB = 128


def _tiny_gpt2_cuda_bf16() -> torch.nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=VOCAB,
        n_positions=SEQ_LEN,
        n_embd=HIDDEN,
        n_layer=N_LAYERS,
        n_head=N_HEAD,
    )
    torch.manual_seed(SEED)
    model = GPT2LMHeadModel(config).to(dtype=torch.bfloat16, device="cuda:0")
    model.eval()
    return model


@pytest.mark.gpu
def test_residual_post_matches_naive_on_cuda_bf16() -> None:
    from fpwap import Callback, Sweep
    from fpwap.types import BatchResult, HookName

    model = _tiny_gpt2_cuda_bf16()

    torch.manual_seed(SEED + 1)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN), device="cuda:0")

    baseline: dict[int, torch.Tensor] = {}

    def _make_hook(layer_idx: int):
        def hook(_mod, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            baseline[layer_idx] = hidden.detach().clone()

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
                i: torch.zeros(
                    N_SAMPLES, SEQ_LEN, HIDDEN, dtype=torch.bfloat16, device="cuda:0"
                )
                for i in range(N_LAYERS)
            }

        def on_batch(
            self,
            layer_idx: int,
            hook: HookName,
            acts: torch.Tensor,
            sample_ids: torch.Tensor,
        ) -> BatchResult:
            self.acts[layer_idx][sample_ids] = acts.detach()
            return None

    cap = Capture()
    run = Sweep(
        model=model,
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[cap],
        transport_dtype=torch.bfloat16,
        seed=SEED,
        apply_final_norm=False,
    )
    result = run.run()

    for layer_idx, base in baseline.items():
        got = cap.acts[layer_idx]
        max_diff = (got.float() - base.float()).abs().max().item()
        assert torch.allclose(got, base, atol=5e-3, rtol=5e-3), (
            f"residual_post mismatch at layer {layer_idx}: max abs diff {max_diff}"
        )

    # Profile recorded per-layer timings even though streaming was a no-op
    # (pre-loaded model path).
    assert len(result.profile.per_layer) == N_LAYERS
