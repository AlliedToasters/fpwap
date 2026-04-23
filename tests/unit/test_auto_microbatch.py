"""Unit tests for estimate_max_microbatch — CI-safe, no GPU required."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from fpwap.engine import estimate_max_microbatch


def _make_config(
    hidden_size: int = 4096,
    intermediate_size: int = 11008,
    num_attention_heads: int = 32,
    vocab_size: int = 32000,
) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        vocab_size=vocab_size,
    )


def test_cpu_returns_n_samples() -> None:
    config = _make_config()
    result = estimate_max_microbatch(
        config=config,
        n_samples=1024,
        seq_len=128,
        transport_dtype=torch.bfloat16,
        exec_device=torch.device("cpu"),
    )
    assert result == 1024


def test_missing_hidden_size_returns_n_samples() -> None:
    config = SimpleNamespace()
    result = estimate_max_microbatch(
        config=config,
        n_samples=512,
        seq_len=128,
        transport_dtype=torch.bfloat16,
        exec_device=torch.device("cpu"),
    )
    assert result == 512


def test_none_config_returns_n_samples() -> None:
    result = estimate_max_microbatch(
        config=None,
        n_samples=256,
        seq_len=64,
        transport_dtype=torch.float32,
        exec_device=torch.device("cpu"),
    )
    assert result == 256


def test_result_is_power_of_two_on_gpu() -> None:
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")

    config = _make_config()
    result = estimate_max_microbatch(
        config=config,
        n_samples=10000,
        seq_len=128,
        transport_dtype=torch.bfloat16,
        exec_device=torch.device("cuda:0"),
    )
    assert result >= 1
    assert result & (result - 1) == 0 or result == 10000


def test_capped_at_n_samples_on_gpu() -> None:
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")

    config = _make_config(hidden_size=64, intermediate_size=256,
                          num_attention_heads=2, vocab_size=100)
    result = estimate_max_microbatch(
        config=config,
        n_samples=4,
        seq_len=16,
        transport_dtype=torch.bfloat16,
        exec_device=torch.device("cuda:0"),
    )
    assert result == 4


def test_smaller_for_larger_model() -> None:
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")

    device = torch.device("cuda:0")
    small = estimate_max_microbatch(
        config=_make_config(hidden_size=1024, intermediate_size=4096,
                            num_attention_heads=8, vocab_size=32000),
        n_samples=4096,
        seq_len=128,
        transport_dtype=torch.bfloat16,
        exec_device=device,
    )
    large = estimate_max_microbatch(
        config=_make_config(hidden_size=8192, intermediate_size=28672,
                            num_attention_heads=64, vocab_size=128256),
        n_samples=4096,
        seq_len=128,
        transport_dtype=torch.bfloat16,
        exec_device=device,
    )
    assert large <= small


def test_buf_on_gpu_reduces_budget() -> None:
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")

    device = torch.device("cuda:0")
    config = _make_config()
    with_buf = estimate_max_microbatch(
        config=config,
        n_samples=1024,
        seq_len=128,
        transport_dtype=torch.bfloat16,
        exec_device=device,
        buf_on_gpu=True,
    )
    without_buf = estimate_max_microbatch(
        config=config,
        n_samples=1024,
        seq_len=128,
        transport_dtype=torch.bfloat16,
        exec_device=device,
        buf_on_gpu=False,
    )
    assert with_buf <= without_buf


def test_safety_factor_affects_result() -> None:
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")

    device = torch.device("cuda:0")
    config = _make_config()
    conservative = estimate_max_microbatch(
        config=config,
        n_samples=4096,
        seq_len=128,
        transport_dtype=torch.bfloat16,
        exec_device=device,
        safety_factor=0.5,
    )
    aggressive = estimate_max_microbatch(
        config=config,
        n_samples=4096,
        seq_len=128,
        transport_dtype=torch.bfloat16,
        exec_device=device,
        safety_factor=0.95,
    )
    assert conservative <= aggressive


def test_intermediate_fallback() -> None:
    """When config has no intermediate_size, defaults to 4*hidden."""
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")

    config_with = _make_config(hidden_size=4096, intermediate_size=16384)
    config_without = SimpleNamespace(
        hidden_size=4096,
        num_attention_heads=32,
        vocab_size=32000,
    )
    device = torch.device("cuda:0")
    result_with = estimate_max_microbatch(
        config=config_with, n_samples=1024, seq_len=128,
        transport_dtype=torch.bfloat16, exec_device=device,
    )
    result_without = estimate_max_microbatch(
        config=config_without, n_samples=1024, seq_len=128,
        transport_dtype=torch.bfloat16, exec_device=device,
    )
    assert result_with == result_without
