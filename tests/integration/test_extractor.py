"""Extractor handle: reuse empty_model + accel_index across sweeps.

Proves that `Extractor.from_hf` builds once and `ext.sweep()` reuses the
pre-built model and index — no rebuild on the second sweep. Uses a tiny
GPT-2 snapshot saved to a temp dir; runs on CPU, no GPU required.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

SEED = 0
N_SAMPLES = 4
SEQ_LEN = 6
HIDDEN = 16
N_LAYERS = 2
N_HEAD = 2
VOCAB = 32


def _write_tiny_gpt2_snapshot(snapshot_dir: Path) -> torch.nn.Module:
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
    return src


def _naive_per_layer_residual_post(
    model: torch.nn.Module, input_ids: torch.Tensor
) -> dict[int, torch.Tensor]:
    captures: dict[int, torch.Tensor] = {}

    def _make_hook(layer_idx: int):
        def hook(_mod, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            captures[layer_idx] = hidden.detach().clone()

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
    return captures


@pytest.mark.integration
def test_extractor_reuses_model_across_sweeps(tmp_path: Path) -> None:
    from fpwap import Callback, Extractor
    from fpwap.types import BatchResult, HookName

    snapshot_dir = tmp_path / "snapshot"
    src = _write_tiny_gpt2_snapshot(snapshot_dir)

    torch.manual_seed(SEED + 1)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    baseline = _naive_per_layer_residual_post(src, input_ids)

    ext = Extractor.from_hf(str(snapshot_dir), dtype=torch.float32)
    model_id_first = id(ext._model)

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
            self.acts[layer_idx][sample_ids] = acts.detach().float().cpu()
            return None

    results = []
    for _ in range(2):
        cap = Capture()
        sweep = ext.sweep(
            dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
            seq_len=SEQ_LEN,
            callbacks=[cap],
            transport_dtype=torch.float32,
            execution_device="cpu",
            seed=SEED,
            apply_final_norm=False,
            progress=False,
        )
        sweep.run()
        results.append(cap.acts)

    assert id(ext._model) == model_id_first, "model object was rebuilt"

    for layer_idx, baseline_hidden in baseline.items():
        for run_idx, acts in enumerate(results):
            got = acts[layer_idx]
            max_diff = (got - baseline_hidden).abs().max().item()
            assert torch.allclose(got, baseline_hidden, atol=1e-5, rtol=1e-5), (
                f"sweep {run_idx} layer {layer_idx} mismatch: max abs diff {max_diff}"
            )

    for layer_idx in range(N_LAYERS):
        assert torch.equal(results[0][layer_idx], results[1][layer_idx]), (
            f"sweeps diverged at layer {layer_idx}"
        )


@pytest.mark.integration
def test_extractor_sweep_uses_streaming_path(tmp_path: Path) -> None:
    """Extractor sweeps must use the streaming path (weight I/O > 0)."""
    from fpwap import Extractor
    from fpwap.callbacks.common import RawActivations

    snapshot_dir = tmp_path / "snapshot"
    _write_tiny_gpt2_snapshot(snapshot_dir)

    ext = Extractor.from_hf(str(snapshot_dir), dtype=torch.float32)

    torch.manual_seed(SEED + 1)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    raw = RawActivations(layers="all", last_token_only=False, out_dtype=torch.float32)
    sweep = ext.sweep(
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[raw],
        transport_dtype=torch.float32,
        execution_device="cpu",
        seed=SEED,
        apply_final_norm=False,
        progress=False,
    )
    result = sweep.run()

    assert result.profile.bytes_moved()["weights"] > 0, (
        "Extractor sweep should use streaming path (non-zero weight I/O)"
    )


@pytest.mark.integration
def test_extractor_sweep_forwards_padding(tmp_path: Path) -> None:
    """Extractor.sweep(padding='bucketed') forwards the kwarg to Sweep."""
    from fpwap import Extractor
    from fpwap.callbacks.common import RawActivations

    snapshot_dir = tmp_path / "snapshot"
    _write_tiny_gpt2_snapshot(snapshot_dir)

    ext = Extractor.from_hf(str(snapshot_dir), dtype=torch.float32)

    torch.manual_seed(SEED + 1)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    raw = RawActivations(layers="all", last_token_only=False, out_dtype=torch.float32)
    sweep = ext.sweep(
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[raw],
        transport_dtype=torch.float32,
        execution_device="cpu",
        seed=SEED,
        apply_final_norm=False,
        progress=False,
        padding="bucketed",
    )
    assert sweep.padding == "bucketed"


@pytest.mark.integration
def test_extractor_sweep_forwards_buffer_path(tmp_path: Path) -> None:
    """Extractor.sweep(buffer_path=...) forwards the kwarg to Sweep."""
    from fpwap import Extractor
    from fpwap.callbacks.common import RawActivations

    snapshot_dir = tmp_path / "snapshot"
    _write_tiny_gpt2_snapshot(snapshot_dir)

    ext = Extractor.from_hf(str(snapshot_dir), dtype=torch.float32)

    torch.manual_seed(SEED + 1)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    raw = RawActivations(layers="all", last_token_only=False, out_dtype=torch.float32)
    buf = tmp_path / "residual.bin"
    sweep = ext.sweep(
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[raw],
        transport_dtype=torch.float32,
        execution_device="cpu",
        seed=SEED,
        apply_final_norm=False,
        progress=False,
        buffer_path=buf,
    )
    assert sweep.buffer_path == buf


@pytest.mark.integration
def test_extractor_requires_execution_device(tmp_path: Path) -> None:
    """Extractor.sweep() without execution_device must raise."""
    from fpwap import Extractor
    from fpwap.callbacks.common import RawActivations

    snapshot_dir = tmp_path / "snapshot"
    _write_tiny_gpt2_snapshot(snapshot_dir)

    ext = Extractor.from_hf(str(snapshot_dir), dtype=torch.float32)

    torch.manual_seed(SEED + 1)
    input_ids = torch.randint(0, VOCAB, (2, SEQ_LEN))

    raw = RawActivations(layers="all", last_token_only=False, out_dtype=torch.float32)
    sweep = ext.sweep(
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(2)],
        seq_len=SEQ_LEN,
        callbacks=[raw],
        transport_dtype=torch.float32,
        seed=SEED,
        apply_final_norm=False,
        progress=False,
    )
    with pytest.raises(ValueError, match="execution_device is required"):
        sweep.run()
