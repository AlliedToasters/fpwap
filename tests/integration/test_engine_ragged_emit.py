"""Engine-level: ragged Emit (#65) flows through the dispatch loop end-to-end.

Verifies that a callback returning Emit(tensor=flat, sample_lengths=lens)
produces a RaggedTensor on result.activations(...), bit-equal to the dense
emit's per-sample slices, both for the in-memory path and the MemmapBackend
path. Pad block must be bypassed (no zero-fill on the variable-length axis).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from fpwap import Callback, Emit
from fpwap.types import HookName

SEED = 0
N_SAMPLES = 6
SEQ_LEN = 8
HIDDEN = 16
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


def _per_sample_keep_lengths(n: int, seq: int) -> list[int]:
    """Deterministic keep-count per sample: 1, 2, 3, ... clipped to seq."""
    return [min(i + 1, seq) for i in range(n)]


class _RaggedKeepFromTail(Callback):
    """Callback: keep the last `keep[i]` tokens of sample i, ragged."""

    def __init__(self, keep: list[int]) -> None:
        self.keep = keep

    def on_batch(
        self,
        layer_idx: int,
        hook: HookName,
        acts: Tensor,
        sample_ids: Tensor,
    ) -> Emit:
        sids = sample_ids.detach().cpu().tolist()
        chunks = []
        lens = []
        for row, sid in enumerate(sids):
            k = self.keep[sid]
            chunks.append(acts[row, -k:, :].detach().to(torch.float32).cpu())
            lens.append(k)
        flat = torch.cat(chunks, dim=0)
        return Emit(
            tensor=flat,
            sample_lengths=torch.tensor(lens, dtype=torch.int64),
        )


@pytest.mark.integration
def test_ragged_emit_in_memory() -> None:
    """In-memory path: Result.activations returns RaggedTensor with correct slices."""
    from fpwap import RaggedTensor, Sweep
    from fpwap.callbacks.common import RawActivations

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    dataset = [{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)]

    keep = _per_sample_keep_lengths(N_SAMPLES, SEQ_LEN)

    # Ground truth: dense emit via RawActivations.
    dense = (
        Sweep(
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
        )
        .run()
        .activations(layer=1, hook="residual_post")
    )
    assert isinstance(dense, torch.Tensor)
    assert dense.shape == (N_SAMPLES, SEQ_LEN, HIDDEN)

    # Ragged path with same callback semantics.
    rag_result = Sweep(
        model=model,
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[_RaggedKeepFromTail(keep)],
        transport_dtype=torch.float32,
        microbatch_size=2,
        seed=SEED,
        progress=False,
    ).run()
    rt = rag_result.activations(layer=1, hook="residual_post")
    assert isinstance(rt, RaggedTensor)
    assert len(rt) == N_SAMPLES
    assert torch.equal(rt.lengths, torch.tensor(keep, dtype=torch.int64))
    for i, k in enumerate(keep):
        assert torch.equal(rt[i], dense[i, -k:, :])


@pytest.mark.integration
def test_ragged_emit_memmap_backend(tmp_path: Path) -> None:
    """Disk path: MemmapBackend round-trips ragged Emit through .bin + sidecar."""
    from fpwap import RaggedTensor, Sweep
    from fpwap.callbacks.common import RawActivations
    from fpwap.storage.memmap import MemmapBackend

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    dataset = [{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)]

    keep = _per_sample_keep_lengths(N_SAMPLES, SEQ_LEN)

    # Dense ground truth (in memory).
    dense = (
        Sweep(
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
        )
        .run()
        .activations(layer=0, hook="residual_post")
    )
    assert isinstance(dense, torch.Tensor)

    backend = MemmapBackend(root=tmp_path)
    result = Sweep(
        model=model,
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[_RaggedKeepFromTail(keep)],
        transport_dtype=torch.float32,
        microbatch_size=2,
        seed=SEED,
        progress=False,
        storage=backend,
    ).run()
    rt = result.activations(layer=0, hook="residual_post")
    assert isinstance(rt, RaggedTensor)
    assert len(rt) == N_SAMPLES
    assert torch.equal(rt.lengths, torch.tensor(keep, dtype=torch.int64))
    for i, k in enumerate(keep):
        assert torch.equal(rt[i], dense[i, -k:, :])

    # Sidecar JSON records ragged layout — file is self-describing.
    import json

    metas = list(tmp_path.glob("*.json"))
    assert len(metas) >= 1
    for m in metas:
        meta = json.loads(m.read_text())
        assert meta.get("layout") == "ragged"
