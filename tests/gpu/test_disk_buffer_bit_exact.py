"""GPU bit-exact: disk-backed ResidualBuffer vs naive forward.

SPEC §3.4 calls for NVMe-backed memmap on the residual buffer so the
inter-layer transport can exceed CPU RAM. This test proves the disk path
produces bit-identical residual_post to a naive HF forward — same contract
as the in-memory tests, different buffer backend.
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
def test_disk_buffer_matches_naive_on_cuda_bf16(tmp_path: object) -> None:
    from pathlib import Path

    from fpwap import Callback, Sweep
    from fpwap.types import BatchResult, HookName

    buf_path = Path(str(tmp_path)) / "residual.bin"

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
        buffer_device="cpu",
        buffer_path=buf_path,
    )
    result = run.run()

    for layer_idx, base in baseline.items():
        got = cap.acts[layer_idx]
        max_diff = (got.float() - base.float()).abs().max().item()
        assert torch.allclose(got, base, atol=5e-3, rtol=5e-3), (
            f"disk-buffer mismatch at layer {layer_idx}: max abs diff {max_diff}"
        )

    assert len(result.profile.per_layer) == N_LAYERS
    assert buf_path.exists(), "memmap file should have been created"


@pytest.mark.gpu
def test_disk_buffer_flush_and_close(tmp_path: object) -> None:
    """Buffer flushes writes to disk and cleans up on close."""
    from pathlib import Path

    from fpwap.buffer import ResidualBuffer

    buf_path = Path(str(tmp_path)) / "buf.bin"
    buf = ResidualBuffer(
        n_samples=4,
        seq_len=8,
        hidden=16,
        dtype=torch.bfloat16,
        device="cpu",
        path=buf_path,
    )

    data = torch.randn(2, 8, 16, dtype=torch.bfloat16)
    buf.write_slice(0, 2, data)
    buf.flush()
    assert buf_path.exists()

    got = buf.read_slice(0, 2)
    assert torch.equal(got, data)

    buf.close()
