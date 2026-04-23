"""Coverage for the safetensors -> torch dtype translation table (SPEC Appendix D.1).

This table is populated in the skeleton, so these tests are expected to pass
on day one. They exist to lock the contract against regression.
"""
from __future__ import annotations

import pytest

from fpwap.loader import _SAFE_TO_TORCH_DTYPE


@pytest.mark.parametrize(
    "safe_name,torch_attr",
    [
        ("F64", "float64"),
        ("F32", "float32"),
        ("F16", "float16"),
        ("BF16", "bfloat16"),
        ("I64", "int64"),
        ("I32", "int32"),
        ("I16", "int16"),
        ("I8", "int8"),
        ("U8", "uint8"),
        ("BOOL", "bool"),
    ],
)
def test_safe_to_torch_dtype_mapping(safe_name: str, torch_attr: str) -> None:
    assert _SAFE_TO_TORCH_DTYPE[safe_name] == torch_attr


def test_all_values_are_real_torch_attrs() -> None:
    import torch

    for safe_name, torch_attr in _SAFE_TO_TORCH_DTYPE.items():
        assert hasattr(torch, torch_attr), (
            f"{safe_name} -> {torch_attr} is not a valid torch attribute"
        )
