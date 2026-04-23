"""GPU execution, CPU-resident residual buffer.

This is the topology that unlocks N_samples × seq × hidden bigger than VRAM
(the whole point of cooking on a single consumer GPU). Fwd runs on CUDA,
buffer is host RAM, transfers every microbatch.

Must still match a naive CUDA forward bit-for-bit within bf16 tolerance.
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


@pytest.mark.gpu
def test_cpu_buffer_gpu_exec_matches_naive() -> None:
    from transformers import GPT2Config, GPT2LMHeadModel

    from fpwap import Callback, Sweep
    from fpwap.types import BatchResult, HookName

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

    torch.manual_seed(SEED + 1)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN), device="cuda:0")

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
            # sample_ids live on exec device (cuda)
            self.acts[layer_idx][sample_ids] = acts.detach()
            return None

    cap = Capture()
    run = Sweep(
        model=model,
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[cap],
        transport_dtype=torch.bfloat16,
        microbatch_size=4,
        buffer_device="cpu",  # KEY: buffer on host RAM
        seed=SEED,
    )
    run.run()

    for layer_idx, base in baseline.items():
        got = cap.acts[layer_idx]
        max_diff = (got.float() - base.float()).abs().max().item()
        assert torch.allclose(got, base, atol=5e-3, rtol=5e-3), (
            f"CPU-buffer/GPU-exec mismatch at layer {layer_idx}: max abs diff {max_diff}"
        )
