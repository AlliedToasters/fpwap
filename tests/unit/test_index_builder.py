"""Contract for build_accel_index_from_hf_cache (SPEC Appendix C)."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from fpwap.loader import build_accel_index_from_hf_cache


def test_build_index_from_tiny_snapshot(tmp_path: Path) -> None:
    shard = {
        "model.embed_tokens.weight": torch.zeros(4, 8, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 8, dtype=torch.bfloat16),
    }
    shard_file = "model-00001-of-00001.safetensors"
    save_file(shard, tmp_path / shard_file)

    hf_index = {
        "metadata": {"total_size": 0},
        "weight_map": {name: shard_file for name in shard},
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(hf_index))

    accel_index = build_accel_index_from_hf_cache(tmp_path)

    for name, tensor in shard.items():
        entry = accel_index[name]
        assert entry["weight_name"] == name
        assert entry["safetensors_file"] == str(tmp_path / shard_file)
        assert entry["dtype"] == "bfloat16"
        assert entry["shape"] == list(tensor.shape)
