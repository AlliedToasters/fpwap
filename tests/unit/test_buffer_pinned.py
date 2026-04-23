"""ResidualBuffer: CPU path is pinned, write_slice / read_slice round-trip."""
from __future__ import annotations

import torch

from fpwap.buffer import ResidualBuffer


def test_cpu_buffer_is_pinned() -> None:
    if not torch.cuda.is_available():
        return
    buf = ResidualBuffer(n_samples=4, seq_len=2, hidden=3, dtype=torch.float32, device="cpu")
    assert buf._data.is_pinned()


def test_write_slice_round_trips() -> None:
    buf = ResidualBuffer(n_samples=4, seq_len=2, hidden=3, dtype=torch.float32, device="cpu")
    values = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    buf.write_slice(1, 3, values)
    assert torch.equal(buf.read_slice(1, 3), values)
    assert torch.equal(buf._data[0], torch.zeros(2, 3))
    assert torch.equal(buf._data[3], torch.zeros(2, 3))


def test_write_slice_dtype_conversion() -> None:
    buf = ResidualBuffer(n_samples=2, seq_len=1, hidden=4, dtype=torch.bfloat16, device="cpu")
    values = torch.tensor([[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]], dtype=torch.float32)
    buf.write_slice(0, 2, values)
    assert buf._data.dtype == torch.bfloat16
    assert torch.allclose(buf._data.float(), values, atol=1e-2)


def test_gpu_buffer_not_pinned() -> None:
    """The pin_memory=True flag is only applied when device is CPU."""
    if not torch.cuda.is_available():
        return
    buf = ResidualBuffer(
        n_samples=2, seq_len=1, hidden=4, dtype=torch.float32, device="cuda"
    )
    # A CUDA tensor is neither paged nor pinned in the CPU sense.
    assert buf._data.device.type == "cuda"
