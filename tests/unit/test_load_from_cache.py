"""Contract for load_from_cache and its split helper (SPEC Appendix C).

CPU execution_device to keep this CI-agnostic.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def _write_tiny_gpt2_snapshot(snapshot_dir: Path) -> None:
    """Make a self-contained snapshot_dir: config.json + shard + index.json.

    Matches HuggingFace's on-disk layout closely enough that
    AutoConfig.from_pretrained(snapshot_dir) and our index-builder both
    work against it, without a network round trip.
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=40,
        n_positions=8,
        n_embd=16,
        n_layer=2,
        n_head=2,
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(snapshot_dir)

    torch.manual_seed(0)
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


def test_build_empty_model_and_index(tmp_path: Path) -> None:
    from fpwap.loader import build_empty_model_and_index

    snapshot_dir = tmp_path / "snapshot"
    _write_tiny_gpt2_snapshot(snapshot_dir)

    model, accel_index, timing = build_empty_model_and_index(
        model_id=str(snapshot_dir),
        snapshot_dir=snapshot_dir,
        dtype=torch.bfloat16,
    )

    # All params are on meta (empty-weights).
    for name, p in model.named_parameters():
        assert p.device.type == "meta", f"{name} not on meta"

    # Tied-weight alias is present (lm_head.weight missing from shard).
    assert "lm_head.weight" in accel_index
    assert "transformer.wte.weight" in accel_index
    assert accel_index["lm_head.weight"] == accel_index["transformer.wte.weight"]

    # Setup timing sub-phases are populated.
    assert timing["config_s"] > 0
    assert timing["model_s"] > 0
    assert timing["index_s"] > 0


def test_load_from_cache_writes_index_before_disk_offload(tmp_path: Path) -> None:
    from fpwap.loader import load_from_cache

    snapshot_dir = tmp_path / "snapshot"
    offload_dir = tmp_path / "offload"
    _write_tiny_gpt2_snapshot(snapshot_dir)

    model = load_from_cache(
        model_id=str(snapshot_dir),
        snapshot_dir=snapshot_dir,
        offload_dir=offload_dir,
        execution_device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    # index.json was written; if it hadn't existed before disk_offload,
    # OffloadedWeightsLoader would have cached an empty index (SPEC D.3).
    assert (offload_dir / "index.json").exists()
    written = json.loads((offload_dir / "index.json").read_text())
    assert "lm_head.weight" in written
    assert "transformer.wte.weight" in written

    # Result is usable (AlignDevicesHooks are installed by disk_offload; fpwap
    # does not rely on them, but a non-fpwap caller could).
    assert hasattr(model, "transformer")
    assert len(model.transformer.h) == 2
