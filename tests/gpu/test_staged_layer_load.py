"""Contract for StagedLayerLoader (pinned-staging single-H2D layer loads).

Same shard-on-disk setup as tests/unit/test_loader_layer.py, but loads
through _OffloadStreamer on CUDA — which routes through StagedLayerLoader —
and checks bit-exactness against the source weights, meta restoration on
unload, reload after unload (pinned-buffer + copy-event reuse), and the
FPWAP_STAGED_LOAD=0 kill switch.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from fpwap.engine import _OffloadStreamer
from fpwap.loader import (
    alias_tied_weights_in_index,
    build_accel_index_from_hf_cache,
)
from fpwap.models import get_plumbing

pytestmark = pytest.mark.gpu

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


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


def _make_streamer(tmp_path: Path) -> tuple[
    torch.nn.Module, torch.nn.Module, _OffloadStreamer
]:
    src, dst = _write_tiny_gpt2_shard(tmp_path)
    accel_index = build_accel_index_from_hf_cache(tmp_path)
    alias_tied_weights_in_index(dst, accel_index)
    streamer = _OffloadStreamer(accel_index, torch.device("cuda"), model=dst)
    return src, dst, streamer


@requires_cuda
def test_staged_load_matches_source_bit_exact(tmp_path: Path) -> None:
    src, dst, streamer = _make_streamer(tmp_path)
    plumbing = get_plumbing(dst)
    assert streamer._staged is not None, "staged path should be active on CUDA"

    handle = streamer.load_layer(dst, 0, plumbing)
    assert handle is not None, "staged load should return a readiness event"
    streamer.wait_load_ready(handle)
    torch.cuda.synchronize()

    src_layer = plumbing.layer_modules(src)[0]
    dst_layer = plumbing.layer_modules(dst)[0]
    for (src_name, src_p), (dst_name, dst_p) in zip(
        src_layer.named_parameters(), dst_layer.named_parameters(), strict=True
    ):
        assert src_name == dst_name
        assert type(dst_p) is torch.nn.Parameter
        assert dst_p.device.type == "cuda", f"{dst_name} not on cuda"
        assert dst_p.requires_grad == src_p.requires_grad
        assert torch.equal(src_p.cuda(), dst_p), f"value mismatch at {dst_name}"
    streamer.close()


@requires_cuda
def test_staged_unload_and_reload(tmp_path: Path) -> None:
    src, dst, streamer = _make_streamer(tmp_path)
    plumbing = get_plumbing(dst)

    streamer.wait_load_ready(streamer.load_layer(dst, 0, plumbing))
    streamer.unload_layer(dst, 0, plumbing)
    dst_layer = plumbing.layer_modules(dst)[0]
    for name, p in dst_layer.named_parameters():
        assert p.device.type == "meta", f"{name} not back on meta after unload"

    # Reload exercises pinned-buffer reuse + the copy-event wait path.
    streamer.wait_load_ready(streamer.load_layer(dst, 0, plumbing))
    torch.cuda.synchronize()
    src_layer = plumbing.layer_modules(src)[0]
    for (name, src_p), (_, dst_p) in zip(
        src_layer.named_parameters(), dst_layer.named_parameters(), strict=True
    ):
        assert torch.equal(src_p.cuda(), dst_p), f"reload mismatch at {name}"
    streamer.close()


@requires_cuda
def test_staged_forward_bit_exact_vs_legacy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same device, same kernels — only the load path differs."""
    torch.manual_seed(1)
    hs = torch.randn(2, 8, 16, device="cuda")

    monkeypatch.setenv("FPWAP_STAGED_LOAD", "0")
    _, legacy_dst, legacy_streamer = _make_streamer(tmp_path)
    legacy_plumbing = get_plumbing(legacy_dst)
    legacy_streamer.wait_load_ready(legacy_streamer.load_layer(legacy_dst, 0, legacy_plumbing))
    with torch.no_grad():
        ref = legacy_plumbing.layer_modules(legacy_dst)[0](hs)[0]
    legacy_streamer.close()

    monkeypatch.delenv("FPWAP_STAGED_LOAD")
    _, dst, streamer = _make_streamer(tmp_path)
    plumbing = get_plumbing(dst)
    assert streamer._staged is not None
    streamer.wait_load_ready(streamer.load_layer(dst, 0, plumbing))
    with torch.no_grad():
        got = plumbing.layer_modules(dst)[0](hs)[0]
    torch.cuda.synchronize()
    assert torch.equal(ref, got), "staged forward diverged from legacy path"
    streamer.close()


@requires_cuda
def test_kill_switch_disables_staged_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPWAP_STAGED_LOAD", "0")
    src, dst, streamer = _make_streamer(tmp_path)
    plumbing = get_plumbing(dst)
    assert streamer._staged is None

    handle = streamer.load_layer(dst, 0, plumbing)
    assert handle is None
    streamer.wait_load_ready(handle)  # no-op
    src_layer = plumbing.layer_modules(src)[0]
    dst_layer = plumbing.layer_modules(dst)[0]
    for (name, src_p), (_, dst_p) in zip(
        src_layer.named_parameters(), dst_layer.named_parameters(), strict=True
    ):
        assert torch.equal(src_p.cuda(), dst_p), f"legacy-path mismatch at {name}"
    streamer.close()


def test_cpu_streamer_has_no_staged_loader(tmp_path: Path) -> None:
    """CPU execution keeps the legacy path (runs in CI without a GPU)."""
    _, dst = _write_tiny_gpt2_shard(tmp_path)
    accel_index = build_accel_index_from_hf_cache(tmp_path)
    alias_tied_weights_in_index(dst, accel_index)
    streamer = _OffloadStreamer(accel_index, torch.device("cpu"), model=dst)
    assert streamer._staged is None
    plumbing = get_plumbing(dst)
    assert streamer.load_layer(dst, 0, plumbing) is None
    streamer.close()
