"""Contract for _load_layer / _unload_layer (SPEC §12.4 Approach A).

Exercises the manual streaming path: construct a real safetensors-backed
shard on disk, build the accelerate index from it, and confirm
_load_layer materializes one block's params from meta → real, matching
the source values bit-exact, while _unload_layer returns them to meta.

CPU-only; no AlignDevicesHook involved.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from fpwap.loader import (
    _load_layer,
    _unload_layer,
    alias_tied_weights_in_index,
    build_accel_index_from_hf_cache,
)
from fpwap.models import get_plumbing


def _write_tiny_gpt2_shard(tmp_path: Path) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Save a tiny GPT-2's state_dict to safetensors under tmp_path.

    Returns (source_model_with_real_weights, empty_twin_on_meta_device).
    """
    from accelerate import init_empty_weights
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=40,
        n_positions=8,
        n_embd=16,
        n_layer=2,
        n_head=2,
    )
    torch.manual_seed(0)
    src = GPT2LMHeadModel(config)
    src.eval()

    # HF strips tied aliases before saving — safetensors refuses shared storage.
    # GPT-2 ties lm_head.weight ↔ transformer.wte.weight. Drop the tied alias
    # from the shard; alias_tied_weights_in_index re-adds it at load time.
    tied_alias = "lm_head.weight"
    state_dict = {
        k: v.contiguous() for k, v in src.state_dict().items() if k != tied_alias
    }
    save_file(state_dict, tmp_path / "model.safetensors")
    hf_index = {
        "metadata": {"total_size": 0},
        "weight_map": {k: "model.safetensors" for k in state_dict},
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(hf_index))

    with init_empty_weights():
        dst = GPT2LMHeadModel(config)
    dst.tie_weights()
    return src, dst


def test_load_layer_matches_source(tmp_path: Path) -> None:
    from accelerate.utils import OffloadedWeightsLoader

    src, dst = _write_tiny_gpt2_shard(tmp_path)
    accel_index = build_accel_index_from_hf_cache(tmp_path)
    alias_tied_weights_in_index(dst, accel_index)
    loader = OffloadedWeightsLoader(index=accel_index)
    plumbing = get_plumbing(dst)

    _load_layer(dst, 0, plumbing, loader, torch.device("cpu"))

    src_layer = plumbing.layer_modules(src)[0]
    dst_layer = plumbing.layer_modules(dst)[0]
    for (src_name, src_p), (dst_name, dst_p) in zip(
        src_layer.named_parameters(), dst_layer.named_parameters(), strict=True
    ):
        assert src_name == dst_name, f"param name mismatch: {src_name} vs {dst_name}"
        assert dst_p.device.type != "meta", f"{dst_name} still on meta after load"
        assert torch.equal(src_p, dst_p), f"value mismatch at {dst_name}"


def test_unload_layer_returns_to_meta(tmp_path: Path) -> None:
    from accelerate.utils import OffloadedWeightsLoader

    _, dst = _write_tiny_gpt2_shard(tmp_path)
    accel_index = build_accel_index_from_hf_cache(tmp_path)
    alias_tied_weights_in_index(dst, accel_index)
    loader = OffloadedWeightsLoader(index=accel_index)
    plumbing = get_plumbing(dst)

    _load_layer(dst, 0, plumbing, loader, torch.device("cpu"))
    _unload_layer(dst, 0, plumbing)

    dst_layer = plumbing.layer_modules(dst)[0]
    for name, p in dst_layer.named_parameters():
        assert p.device.type == "meta", f"{name} not on meta after unload"
