"""MemmapBackend round-trip: emits go to disk, activations() reads them back.

Contract: when a sweep is given a MemmapBackend, `result.activations(layer,
hook)` returns the same tensor as the in-memory path would, but backed by
a numpy memmap on disk (no RAM pressure for the full [N, S, H] corpus).
"""
from __future__ import annotations

from pathlib import Path

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
def test_memmap_backend_matches_in_memory(tmp_path: Path) -> None:
    from fpwap import Sweep
    from fpwap.callbacks.common import RawActivations
    from fpwap.storage.memmap import MemmapBackend

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    dataset = [{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)]

    def make_sweep(storage):
        return Sweep(
            model=model,
            dataset=dataset,
            seq_len=SEQ_LEN,
            callbacks=[
                RawActivations(layers="all", last_token_only=False, out_dtype=torch.float32)
            ],
            transport_dtype=torch.float32,
            microbatch_size=2,
            seed=SEED,
            progress=False,
            storage=storage,
        )

    # Ground truth: in-memory path
    result_mem = make_sweep(None).run()

    # Disk path
    backend = MemmapBackend(root=tmp_path)
    result_disk = make_sweep(backend).run()

    for layer_idx in range(N_LAYERS):
        mem = result_mem.activations(layer=layer_idx, hook="residual_post")
        disk = result_disk.activations(layer=layer_idx, hook="residual_post")
        assert disk.shape == mem.shape == (N_SAMPLES, SEQ_LEN, HIDDEN)
        assert torch.equal(disk, mem), (
            f"layer {layer_idx}: max diff {(disk - mem).abs().max().item()}"
        )

    # Files exist on disk
    files = list(tmp_path.glob("*.bin"))
    assert len(files) == N_LAYERS


@pytest.mark.integration
def test_memmap_backend_pooled_shape(tmp_path: Path) -> None:
    """Pooled (last_token_only) path shape: [N, H]."""
    from fpwap import Sweep
    from fpwap.callbacks.common import RawActivations
    from fpwap.storage.memmap import MemmapBackend

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    dataset = [{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)]

    backend = MemmapBackend(root=tmp_path)
    sweep = Sweep(
        model=model,
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[
            RawActivations(layers=[1], last_token_only=True, out_dtype=torch.float32)
        ],
        transport_dtype=torch.float32,
        microbatch_size=3,
        seed=SEED,
        progress=False,
        storage=backend,
    )
    result = sweep.run()
    got = result.activations(layer=1, hook="residual_post")
    assert got.shape == (N_SAMPLES, HIDDEN)


@pytest.mark.integration
def test_memmap_backend_bf16_roundtrip(tmp_path: Path) -> None:
    """bf16 emits: MemmapBackend stores bit-pattern as uint16 since numpy has
    no native bf16. Round-trip through disk must preserve torch.equal.
    """
    from fpwap import Sweep
    from fpwap.callbacks.common import RawActivations
    from fpwap.storage.memmap import MemmapBackend

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    dataset = [{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)]

    def make_sweep(storage):
        return Sweep(
            model=model,
            dataset=dataset,
            seq_len=SEQ_LEN,
            callbacks=[
                RawActivations(layers=[0], last_token_only=True, out_dtype=torch.bfloat16)
            ],
            transport_dtype=torch.float32,
            microbatch_size=N_SAMPLES,
            seed=SEED,
            progress=False,
            storage=storage,
        )

    mem = make_sweep(None).run().activations(layer=0, hook="residual_post")
    disk = make_sweep(MemmapBackend(root=tmp_path)).run().activations(
        layer=0, hook="residual_post"
    )
    assert mem.dtype == torch.bfloat16
    assert disk.dtype == torch.bfloat16
    assert torch.equal(disk, mem)
