"""Integration tests for multi-layer chunk processing.

Verifies that chunk_size > 1 produces bit-identical results to chunk_size=1,
that callbacks fire per-layer within chunks, and that profiling reports
all layers correctly.

Uses a 4-layer tiny GPT-2 on CPU/fp32 for determinism.
"""
from __future__ import annotations

import pytest
import torch

SEED = 0
N_SAMPLES = 6
SEQ_LEN = 8
HIDDEN = 32
N_LAYERS = 4
N_HEAD = 2
VOCAB = 100


def _tiny_gpt2_4layer() -> torch.nn.Module:
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


def _make_dataset() -> list[dict]:
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    return [{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)]


def _capture_callback():
    from fpwap import Callback
    from fpwap.types import BatchResult, HookName

    class Capture(Callback):
        phase = "read"
        target_layers = "all"
        target_hooks: tuple[HookName, ...] = ("residual_post",)

        def __init__(self) -> None:
            self.acts: dict[int, torch.Tensor] = {
                i: torch.zeros(N_SAMPLES, SEQ_LEN, HIDDEN) for i in range(N_LAYERS)
            }
            self.layer_starts: list[int] = []
            self.layer_ends: list[int] = []

        def on_layer_start(self, layer_idx: int) -> None:
            self.layer_starts.append(layer_idx)

        def on_batch(
            self,
            layer_idx: int,
            hook: HookName,
            acts: torch.Tensor,
            sample_ids: torch.Tensor,
        ) -> BatchResult:
            self.acts[layer_idx][sample_ids] = acts.detach().float().cpu()
            return None

        def on_layer_end(self, layer_idx: int) -> None:
            self.layer_ends.append(layer_idx)
            return None

    return Capture()


@pytest.mark.integration
def test_chunk2_matches_chunk1() -> None:
    """chunk_size=2 must produce identical residual_post as chunk_size=1."""
    from fpwap import Sweep

    model = _tiny_gpt2_4layer()
    dataset = _make_dataset()

    cap1 = _capture_callback()
    s1 = Sweep(
        model=model, dataset=dataset, seq_len=SEQ_LEN, callbacks=[cap1],
        transport_dtype=torch.float32, seed=SEED, apply_final_norm=False,
        chunk_size=1,
    )
    s1.run()

    cap2 = _capture_callback()
    s2 = Sweep(
        model=model, dataset=dataset, seq_len=SEQ_LEN, callbacks=[cap2],
        transport_dtype=torch.float32, seed=SEED, apply_final_norm=False,
        chunk_size=2,
    )
    s2.run()

    for layer_idx in range(N_LAYERS):
        assert torch.allclose(cap1.acts[layer_idx], cap2.acts[layer_idx], atol=1e-6), (
            f"chunk_size=2 diverged at layer {layer_idx}"
        )


@pytest.mark.integration
def test_chunk_all_layers_matches_chunk1() -> None:
    """chunk_size=n_layers (single chunk) must match chunk_size=1."""
    from fpwap import Sweep

    model = _tiny_gpt2_4layer()
    dataset = _make_dataset()

    cap1 = _capture_callback()
    s1 = Sweep(
        model=model, dataset=dataset, seq_len=SEQ_LEN, callbacks=[cap1],
        transport_dtype=torch.float32, seed=SEED, apply_final_norm=False,
        chunk_size=1,
    )
    s1.run()

    cap_all = _capture_callback()
    s_all = Sweep(
        model=model, dataset=dataset, seq_len=SEQ_LEN, callbacks=[cap_all],
        transport_dtype=torch.float32, seed=SEED, apply_final_norm=False,
        chunk_size=N_LAYERS,
    )
    s_all.run()

    for layer_idx in range(N_LAYERS):
        assert torch.allclose(cap1.acts[layer_idx], cap_all.acts[layer_idx], atol=1e-6), (
            f"chunk_size={N_LAYERS} diverged at layer {layer_idx}"
        )


@pytest.mark.integration
def test_chunk_uneven_division() -> None:
    """4 layers, chunk_size=3 -> chunks [0,1,2] and [3]. Must match chunk_size=1."""
    from fpwap import Sweep

    model = _tiny_gpt2_4layer()
    dataset = _make_dataset()

    cap1 = _capture_callback()
    s1 = Sweep(
        model=model, dataset=dataset, seq_len=SEQ_LEN, callbacks=[cap1],
        transport_dtype=torch.float32, seed=SEED, apply_final_norm=False,
        chunk_size=1,
    )
    s1.run()

    cap3 = _capture_callback()
    s3 = Sweep(
        model=model, dataset=dataset, seq_len=SEQ_LEN, callbacks=[cap3],
        transport_dtype=torch.float32, seed=SEED, apply_final_norm=False,
        chunk_size=3,
    )
    s3.run()

    for layer_idx in range(N_LAYERS):
        assert torch.allclose(cap1.acts[layer_idx], cap3.acts[layer_idx], atol=1e-6), (
            f"chunk_size=3 diverged at layer {layer_idx}"
        )


@pytest.mark.integration
def test_chunk_callbacks_fire_per_layer() -> None:
    """on_layer_start and on_layer_end must fire for every layer, in order."""
    from fpwap import Sweep

    model = _tiny_gpt2_4layer()
    dataset = _make_dataset()

    cap = _capture_callback()
    s = Sweep(
        model=model, dataset=dataset, seq_len=SEQ_LEN, callbacks=[cap],
        transport_dtype=torch.float32, seed=SEED, apply_final_norm=False,
        chunk_size=2,
    )
    s.run()

    assert cap.layer_starts == [0, 1, 2, 3]
    assert cap.layer_ends == [0, 1, 2, 3]


@pytest.mark.integration
def test_chunk_profile_reports_all_layers() -> None:
    """Profile must have per_layer entries for all layer indices."""
    from fpwap import Sweep

    model = _tiny_gpt2_4layer()
    dataset = _make_dataset()

    cap = _capture_callback()
    s = Sweep(
        model=model, dataset=dataset, seq_len=SEQ_LEN, callbacks=[cap],
        transport_dtype=torch.float32, seed=SEED, apply_final_norm=False,
        chunk_size=2,
    )
    result = s.run()

    assert set(result.profile.per_layer.keys()) == {0, 1, 2, 3}
    for timing in result.profile.per_layer.values():
        assert timing.forward_s > 0


@pytest.mark.integration
def test_chunk_exceeds_n_layers() -> None:
    """chunk_size > n_layers behaves identically to chunk_size=n_layers."""
    from fpwap import Sweep

    model = _tiny_gpt2_4layer()
    dataset = _make_dataset()

    cap1 = _capture_callback()
    s1 = Sweep(
        model=model, dataset=dataset, seq_len=SEQ_LEN, callbacks=[cap1],
        transport_dtype=torch.float32, seed=SEED, apply_final_norm=False,
        chunk_size=N_LAYERS,
    )
    s1.run()

    cap_big = _capture_callback()
    s_big = Sweep(
        model=model, dataset=dataset, seq_len=SEQ_LEN, callbacks=[cap_big],
        transport_dtype=torch.float32, seed=SEED, apply_final_norm=False,
        chunk_size=100,
    )
    s_big.run()

    for layer_idx in range(N_LAYERS):
        assert torch.allclose(cap1.acts[layer_idx], cap_big.acts[layer_idx], atol=1e-6)


@pytest.mark.integration
def test_chunk_with_final_norm() -> None:
    """chunk_size > 1 with apply_final_norm=True must still apply norm correctly."""
    from fpwap import Sweep

    model = _tiny_gpt2_4layer()
    dataset = _make_dataset()

    cap1 = _capture_callback()
    s1 = Sweep(
        model=model, dataset=dataset, seq_len=SEQ_LEN, callbacks=[cap1],
        transport_dtype=torch.float32, seed=SEED, apply_final_norm=True,
        chunk_size=1,
    )
    s1.run()

    cap2 = _capture_callback()
    s2 = Sweep(
        model=model, dataset=dataset, seq_len=SEQ_LEN, callbacks=[cap2],
        transport_dtype=torch.float32, seed=SEED, apply_final_norm=True,
        chunk_size=2,
    )
    s2.run()

    last = N_LAYERS - 1
    assert torch.allclose(cap1.acts[last], cap2.acts[last], atol=1e-6), (
        "final norm result differs with chunk_size=2"
    )


@pytest.mark.integration
def test_chunk_reduces_buffer_writes_gpu() -> None:
    """With chunk_size=2 and buf != exec device, first-in-chunk layers skip buffer write.

    The GPU scratch optimization only activates when buffer_device differs from
    execution_device (the real streaming use case). On CPU-only, both are the
    same device, so every layer still writes to the buffer.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA for different buf/exec devices")

    from fpwap import Sweep

    model = _tiny_gpt2_4layer()
    dataset = _make_dataset()
    device = torch.device("cuda:0")
    model = model.to(device)

    cap = _capture_callback()
    s = Sweep(
        model=model, dataset=dataset, seq_len=SEQ_LEN, callbacks=[cap],
        transport_dtype=torch.float32, seed=SEED, apply_final_norm=False,
        chunk_size=2, progress=False,
        execution_device=device, buffer_device="cpu",
    )
    result = s.run()

    # With chunk_size=2: chunks are [0,1] and [2,3].
    # Layer 0: first in chunk → reads from CPU buffer, writes to GPU scratch
    # Layer 1: last in chunk → reads from GPU scratch, writes to CPU buffer
    # Layer 2: first in chunk → reads from CPU buffer, writes to GPU scratch
    # Layer 3: last in chunk → reads from GPU scratch, writes to CPU buffer
    assert result.profile.per_layer[0].bytes_buffer == 0, (
        "layer 0 (first in chunk) should have no buffer write"
    )
    assert result.profile.per_layer[2].bytes_buffer == 0, (
        "layer 2 (first in chunk) should have no buffer write"
    )
    assert result.profile.per_layer[1].bytes_buffer > 0, (
        "layer 1 (last in chunk) should have buffer write"
    )
    assert result.profile.per_layer[3].bytes_buffer > 0, (
        "layer 3 (last in chunk) should have buffer write"
    )
