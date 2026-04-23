"""Integration test for multi-dataset warm-start via Extractor handle.

Proves that running N sweeps back-to-back through the same Extractor
reuses the model object (no rebuild) and produces consistent results.
Uses a tiny GPT-2 snapshot; CPU-only, no GPU required.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from fpwap import Callback, Extractor, Sweep
from fpwap.types import BatchResult, HookName

SEED = 0
N_SAMPLES = 4
SEQ_LEN = 6
HIDDEN = 16
N_LAYERS = 2
N_HEAD = 2
VOCAB = 32
N_SWEEPS = 5


def _write_tiny_gpt2_snapshot(snapshot_dir: Path) -> None:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=VOCAB,
        n_positions=SEQ_LEN,
        n_embd=HIDDEN,
        n_layer=N_LAYERS,
        n_head=N_HEAD,
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(snapshot_dir)

    torch.manual_seed(SEED)
    src = GPT2LMHeadModel(config)
    src.eval()

    state_dict = {
        k: v.contiguous() for k, v in src.state_dict().items() if k != "lm_head.weight"
    }
    save_file(state_dict, snapshot_dir / "model.safetensors")
    hf_index = {
        "metadata": {"total_size": 0},
        "weight_map": {k: "model.safetensors" for k in state_dict},
    }
    (snapshot_dir / "model.safetensors.index.json").write_text(json.dumps(hf_index))


class _TouchCounter(Callback):
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


def _make_dataset() -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(SEED + 1)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    return [{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)]


@pytest.mark.integration
def test_warm_start_reuses_model(tmp_path: Path) -> None:
    """Extractor model object identity is stable across N sweeps."""
    snapshot_dir = tmp_path / "snapshot"
    _write_tiny_gpt2_snapshot(snapshot_dir)

    ext = Extractor.from_hf(str(snapshot_dir), dtype=torch.float32)
    model_id_at_start = id(ext._model)

    dataset = _make_dataset()
    for _ in range(N_SWEEPS):
        cb = _TouchCounter()
        sweep = ext.sweep(
            dataset=dataset,
            seq_len=SEQ_LEN,
            callbacks=[cb],
            transport_dtype=torch.float32,
            execution_device="cpu",
            seed=SEED,
            progress=False,
            apply_final_norm=False,
        )
        sweep.run()
        assert cb.touches == N_SAMPLES * N_LAYERS

    assert id(ext._model) == model_id_at_start, "model was rebuilt during sweeps"


@pytest.mark.integration
def test_warm_start_consistent_profiles(tmp_path: Path) -> None:
    """All N sweeps should report non-zero weight I/O (streaming path)."""
    snapshot_dir = tmp_path / "snapshot"
    _write_tiny_gpt2_snapshot(snapshot_dir)

    ext = Extractor.from_hf(str(snapshot_dir), dtype=torch.float32)
    dataset = _make_dataset()

    weight_bytes_per_sweep: list[int] = []
    for _ in range(N_SWEEPS):
        cb = _TouchCounter()
        sweep = ext.sweep(
            dataset=dataset,
            seq_len=SEQ_LEN,
            callbacks=[cb],
            transport_dtype=torch.float32,
            execution_device="cpu",
            seed=SEED,
            progress=False,
            apply_final_norm=False,
        )
        result = sweep.run()
        weight_bytes_per_sweep.append(result.profile.bytes_moved()["weights"])

    assert all(b > 0 for b in weight_bytes_per_sweep), (
        "warm sweeps should still use streaming path (non-zero weight I/O)"
    )
    assert len(set(weight_bytes_per_sweep)) == 1, (
        f"weight I/O should be identical across sweeps: {weight_bytes_per_sweep}"
    )


@pytest.mark.integration
def test_warm_vs_cold_setup_cost(tmp_path: Path) -> None:
    """Warm-start amortizes setup: N warm sweeps should be faster than N cold."""
    snapshot_dir = tmp_path / "snapshot"
    _write_tiny_gpt2_snapshot(snapshot_dir)
    dataset = _make_dataset()

    t0 = time.perf_counter()
    ext = Extractor.from_hf(str(snapshot_dir), dtype=torch.float32)
    for _ in range(N_SWEEPS):
        cb = _TouchCounter()
        sweep = ext.sweep(
            dataset=dataset,
            seq_len=SEQ_LEN,
            callbacks=[cb],
            transport_dtype=torch.float32,
            execution_device="cpu",
            seed=SEED,
            progress=False,
            apply_final_norm=False,
        )
        sweep.run()
    warm_total = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(N_SWEEPS):
        cb = _TouchCounter()
        sweep = Sweep(
            model=str(snapshot_dir),
            dataset=dataset,
            seq_len=SEQ_LEN,
            callbacks=[cb],
            transport_dtype=torch.float32,
            execution_device="cpu",
            seed=SEED,
            progress=False,
            apply_final_norm=False,
        )
        sweep.run()
    cold_total = time.perf_counter() - t0

    assert warm_total < cold_total, (
        f"warm ({warm_total:.3f}s) should be faster than cold ({cold_total:.3f}s)"
    )
