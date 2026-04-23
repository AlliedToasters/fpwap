"""Unit tests for bucket-building helpers (CI-safe, no GPU)."""
from __future__ import annotations

import torch


def test_next_power_of_2() -> None:
    from fpwap.engine import _next_power_of_2

    assert _next_power_of_2(1) == 1
    assert _next_power_of_2(2) == 2
    assert _next_power_of_2(3) == 4
    assert _next_power_of_2(5) == 8
    assert _next_power_of_2(16) == 16
    assert _next_power_of_2(17) == 32
    assert _next_power_of_2(50) == 64
    assert _next_power_of_2(400) == 512



def test_detect_left_padding() -> None:
    from fpwap.engine import _detect_left_padding

    # Left-padded: mask starts with 0s
    items = [
        {
            "attention_mask": torch.tensor([[0, 0, 0, 1, 1, 1]]),
            "input_ids": torch.tensor([[0, 0, 0, 5, 6, 7]]),
        }
    ]
    assert _detect_left_padding(items) is True

    # Right-padded: mask ends with 0s
    items = [
        {
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0]]),
            "input_ids": torch.tensor([[5, 6, 7, 0, 0, 0]]),
        }
    ]
    assert _detect_left_padding(items) is False

    # No padding present (all 1s) — defaults to True
    items = [
        {
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
        }
    ]
    assert _detect_left_padding(items) is True



def test_trim_to_length_left_padded() -> None:
    from fpwap.engine import _trim_to_length

    item = {
        "input_ids": torch.tensor([[0, 0, 0, 5, 6, 7]]),
        "attention_mask": torch.tensor([[0, 0, 0, 1, 1, 1]]),
        "label": 42,
    }
    trimmed = _trim_to_length(item, 4, left_padded=True)

    assert trimmed["input_ids"].shape == (1, 4)
    assert trimmed["attention_mask"].shape == (1, 4)
    assert torch.equal(trimmed["input_ids"], torch.tensor([[0, 5, 6, 7]]))
    assert torch.equal(trimmed["attention_mask"], torch.tensor([[0, 1, 1, 1]]))
    assert trimmed["label"] == 42  # non-tensor keys pass through



def test_trim_to_length_right_padded() -> None:
    from fpwap.engine import _trim_to_length

    item = {
        "input_ids": torch.tensor([[5, 6, 7, 0, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0]]),
    }
    trimmed = _trim_to_length(item, 4, left_padded=False)

    assert torch.equal(trimmed["input_ids"], torch.tensor([[5, 6, 7, 0]]))
    assert torch.equal(trimmed["attention_mask"], torch.tensor([[1, 1, 1, 0]]))



def test_build_bucketed_segments() -> None:
    from fpwap.engine import _build_bucketed_segments

    max_seq = 64
    hidden = 16
    items = []
    # 3 short (len ~5) + 3 long (len ~40) — should go into different buckets
    for L in [4, 5, 6, 35, 40, 45]:
        ids = torch.full((1, max_seq), 0, dtype=torch.long)
        mask = torch.zeros((1, max_seq), dtype=torch.long)
        ids[0, max_seq - L :] = torch.randint(1, 100, (L,))
        mask[0, max_seq - L :] = 1
        items.append({"input_ids": ids, "attention_mask": mask})

    segments = _build_bucketed_segments(
        items, max_seq, hidden, torch.float32, torch.device("cpu"),
        mb_size_override=6, config=None, exec_device=torch.device("cpu"),
    )

    # Should produce at least 2 segments (short bucket and long bucket)
    assert len(segments) >= 2

    # All original indices covered exactly once
    all_indices = []
    for seg in segments:
        all_indices.extend(seg.orig_indices)
    assert sorted(all_indices) == list(range(len(items)))

    # Each segment's buffer seq_len <= max_seq
    for seg in segments:
        assert seg.seq_len <= max_seq
        assert seg.buffer.seq_len == seg.seq_len
        assert seg.buffer.n_samples == seg.n_samples

    # Short items got a shorter bucket than long items
    short_seg = [s for s in segments if s.seq_len < 32]
    long_seg = [s for s in segments if s.seq_len >= 32]
    assert len(short_seg) >= 1, "expected a short-sequence bucket"
    assert len(long_seg) >= 1, "expected a long-sequence bucket"
