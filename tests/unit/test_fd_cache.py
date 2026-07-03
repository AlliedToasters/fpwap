"""_FdCache: the shard-fd cache must not leak under thread races (#89).

The staged-read pool (and, symmetrically, the advisor's mincore watchdog)
funnels every read through a path → fd cache. Before _FdCache the cache was
an unsynchronized check-then-open dict: on a cold first access every racing
pool thread passed the None check, opened its own fd, and all but the last
writer leaked — invisible to close(), accumulating across loader instances
until EMFILE. The tests here drive the race deterministically by holding
os.open slow while a barrier releases all threads at once.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from fpwap.loader import _FdCache


@pytest.fixture()
def shard(tmp_path):  # type: ignore[no-untyped-def]
    path = tmp_path / "shard.safetensors"
    path.write_bytes(b"\x00" * 64)
    return str(path)


def test_returns_usable_cached_fd(shard: str) -> None:
    cache = _FdCache()
    fd = cache.get(shard)
    assert cache.get(shard) == fd
    assert os.pread(fd, 4, 0) == b"\x00" * 4
    cache.close()
    with pytest.raises(OSError):
        os.fstat(fd)


def test_close_is_idempotent(shard: str) -> None:
    cache = _FdCache()
    cache.get(shard)
    cache.close()
    cache.close()


def test_racing_threads_share_one_fd_and_leak_none(
    shard: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    n_threads = 8
    opened: list[int] = []
    closed: list[int] = []
    record_lock = threading.Lock()
    barrier = threading.Barrier(n_threads)
    real_open, real_close = os.open, os.close

    def slow_open(path: str, flags: int) -> int:
        fd = real_open(path, flags)
        with record_lock:
            opened.append(fd)
        time.sleep(0.05)  # hold the race window open past every thread's check
        return fd

    def tracking_close(fd: int) -> None:
        with record_lock:
            closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(os, "open", slow_open)
    monkeypatch.setattr(os, "close", tracking_close)

    cache = _FdCache()
    results: list[int] = []

    def hit() -> None:
        barrier.wait()
        fd = cache.get(shard)
        with record_lock:
            results.append(fd)

    threads = [threading.Thread(target=hit) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The race actually happened (multiple opens), every caller got the same
    # fd, and after close() no opened fd remains unclosed.
    assert len(opened) > 1, "race never opened concurrently — test is vacuous"
    assert set(results) == {cache.get(shard)}
    cache.close()
    assert sorted(opened) == sorted(closed)
