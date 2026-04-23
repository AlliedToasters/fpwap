"""Contract for alias_tied_weights_in_index (SPEC Appendix C / D.2).

Tied weights (e.g. Llama's lm_head ↔ embed_tokens) only appear once in a
safetensors shard. The loader must detect these via object-identity on
named_parameters(remove_duplicate=False) and add alias entries so hook
lookups by absolute module-parameter path succeed for both names.
"""
from __future__ import annotations

from torch import nn

from fpwap.loader import alias_tied_weights_in_index


class _TiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4, bias=False)
        self.b = nn.Linear(4, 4, bias=False)
        # tie: b.weight shares storage with a.weight
        self.b.weight = self.a.weight


def test_alias_adds_missing_tied_key() -> None:
    model = _TiedModel()
    accel_index: dict[str, dict[str, object]] = {
        "a.weight": {
            "safetensors_file": "/fake/shard.safetensors",
            "weight_name": "a.weight",
            "dtype": "float32",
            "shape": [4, 4],
        }
    }

    alias_tied_weights_in_index(model, accel_index)

    assert "b.weight" in accel_index
    assert accel_index["b.weight"] == accel_index["a.weight"]


def test_alias_is_noop_when_all_names_present() -> None:
    model = _TiedModel()
    entry_a: dict[str, object] = {
        "safetensors_file": "/fake/shard.safetensors",
        "weight_name": "a.weight",
        "dtype": "float32",
        "shape": [4, 4],
    }
    entry_b: dict[str, object] = {**entry_a, "weight_name": "b.weight"}
    accel_index = {"a.weight": entry_a, "b.weight": entry_b}

    alias_tied_weights_in_index(model, accel_index)

    # Both keys still point at their original entries.
    assert accel_index["a.weight"] is entry_a
    assert accel_index["b.weight"] is entry_b


def test_alias_does_nothing_for_untied_params() -> None:
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
    accel_index: dict[str, dict[str, object]] = {
        "0.weight": {
            "safetensors_file": "/fake/shard.safetensors",
            "weight_name": "0.weight",
            "dtype": "float32",
            "shape": [4, 4],
        }
    }

    alias_tied_weights_in_index(model, accel_index)

    assert "1.weight" not in accel_index
