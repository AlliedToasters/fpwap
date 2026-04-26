"""Result.activations(..., as_path=...) — issue #70.

Avoids materializing the emit into a host-RAM tensor when the consumer
already wants a disk handle. Two modes:

* `as_path=True` — return a `ResultArtifact` pointing at the backend's
  in-place file (caller mmap-reads on demand).
* `as_path=Path(dest_dir)` — hardlink (or copy on EXDEV) the data .bin
  and the .json sidecar into `dest_dir`; return a `ResultArtifact`
  pointing at the new location. Backend keeps its original — ownership
  is *not* transferred.

Both modes return the same dataclass shape, so callers can ignore the
distinction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

from fpwap import Callback, Emit
from fpwap.types import HookName

SEED = 0
N_SAMPLES = 6
SEQ_LEN = 8
HIDDEN = 16
N_LAYERS = 2
N_HEAD = 2
VOCAB = 100


def _tiny_gpt2() -> torch.nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=VOCAB,
        n_positions=SEQ_LEN,
        n_embd=HIDDEN,
        n_layer=N_LAYERS,
        n_head=N_HEAD,
    )
    torch.manual_seed(SEED)
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


def _dataset() -> list[dict[str, Tensor]]:
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    return [{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)]


def _per_sample_keep_lengths(n: int, seq: int) -> list[int]:
    return [min(i + 1, seq) for i in range(n)]


class _RaggedKeepFromTail(Callback):
    def __init__(self, keep: list[int]) -> None:
        self.keep = keep

    def on_batch(
        self,
        layer_idx: int,
        hook: HookName,
        acts: Tensor,
        sample_ids: Tensor,
    ) -> Emit:
        sids = sample_ids.detach().cpu().tolist()
        chunks = []
        lens = []
        for row, sid in enumerate(sids):
            k = self.keep[sid]
            chunks.append(acts[row, -k:, :].detach().to(torch.float32).cpu())
            lens.append(k)
        flat = torch.cat(chunks, dim=0)
        return Emit(
            tensor=flat,
            sample_lengths=torch.tensor(lens, dtype=torch.int64),
        )


def _read_dense_memmap(path: Path, sidecar: Path) -> Tensor:
    """Mirror of MemmapBackend's read path, used to verify ResultArtifact
    points at a self-describing on-disk artifact (no fpwap calls)."""
    meta = json.loads(sidecar.read_text())
    per_row = tuple(meta["per_row_shape"])
    n_samples = int(meta["n_samples"])
    bf16_as_u16 = bool(meta.get("bf16_as_u16", False))
    np_dtype = np.uint16 if bf16_as_u16 else np.dtype(meta["dtype"])
    mm = np.memmap(path, dtype=np_dtype, mode="r", shape=(n_samples, *per_row))
    t = torch.from_numpy(np.asarray(mm))
    if bf16_as_u16:
        t = t.view(torch.bfloat16)
    return t


def _read_ragged_memmap(path: Path, sidecar: Path):
    """Reconstruct a RaggedTensor from `(data_path, sidecar_path)` alone.

    Mirrors `_read_dense_memmap` for the ragged layout — same purpose:
    proves the on-disk artifact is self-describing (offsets in sidecar,
    flat tensor in .bin) without going through any fpwap accessor.
    """
    from fpwap import RaggedTensor

    meta = json.loads(sidecar.read_text())
    assert meta["layout"] == "ragged"
    per_row = tuple(meta["per_row_shape"])
    bf16_as_u16 = bool(meta.get("bf16_as_u16", False))
    np_dtype = np.uint16 if bf16_as_u16 else np.dtype(meta["dtype"])
    offsets = torch.tensor(meta["offsets"], dtype=torch.int64)
    total = int(offsets[-1].item())
    mm = np.memmap(path, dtype=np_dtype, mode="r", shape=(total, *per_row))
    flat = torch.from_numpy(np.asarray(mm))
    if bf16_as_u16:
        flat = flat.view(torch.bfloat16)
    return RaggedTensor(flat=flat, offsets=offsets)


@pytest.mark.integration
def test_as_path_dense_inplace(tmp_path: Path) -> None:
    """as_path=True returns a ResultArtifact pointing at the backend's
    file; mmap-reading it gives the same tensor as activations() would."""
    from fpwap import ResultArtifact, Sweep
    from fpwap.callbacks.common import RawActivations
    from fpwap.storage.memmap import MemmapBackend

    model = _tiny_gpt2()
    backend_root = tmp_path / "backend"
    backend = MemmapBackend(root=backend_root)
    result = Sweep(
        model=model,
        dataset=_dataset(),
        seq_len=SEQ_LEN,
        callbacks=[
            RawActivations(layers="all", last_token_only=False, out_dtype=torch.float32)
        ],
        transport_dtype=torch.float32,
        microbatch_size=2,
        seed=SEED,
        progress=False,
        storage=backend,
    ).run()

    art = result.activations(layer=0, hook="residual_post", as_path=True)
    assert isinstance(art, ResultArtifact)
    assert art.layout == "dense"
    assert art.shape == (N_SAMPLES, SEQ_LEN, HIDDEN)
    assert art.dtype == torch.float32
    assert art.data_path.exists()
    assert art.data_path.parent == backend_root
    assert art.sidecar_path is not None and art.sidecar_path.exists()

    # Self-describing: a third party can mmap-read with only the artifact.
    via_path = _read_dense_memmap(art.data_path, art.sidecar_path)
    via_tensor = result.activations(layer=0, hook="residual_post")
    assert torch.equal(via_path, via_tensor)


@pytest.mark.integration
def test_as_path_dense_hardlink_to_dest(tmp_path: Path) -> None:
    """as_path=Path(dest_dir) hardlinks data + sidecar into dest_dir;
    backend keeps ownership of the original."""
    from fpwap import ResultArtifact, Sweep
    from fpwap.callbacks.common import RawActivations
    from fpwap.storage.memmap import MemmapBackend

    model = _tiny_gpt2()
    backend_root = tmp_path / "backend"
    dest = tmp_path / "user_dest"
    dest.mkdir()
    backend = MemmapBackend(root=backend_root)
    result = Sweep(
        model=model,
        dataset=_dataset(),
        seq_len=SEQ_LEN,
        callbacks=[
            RawActivations(layers="all", last_token_only=False, out_dtype=torch.float32)
        ],
        transport_dtype=torch.float32,
        microbatch_size=2,
        seed=SEED,
        progress=False,
        storage=backend,
    ).run()

    art = result.activations(
        layer=1, hook="residual_post", as_path=dest
    )
    assert isinstance(art, ResultArtifact)
    assert art.data_path.parent == dest
    assert art.sidecar_path is not None and art.sidecar_path.parent == dest

    # Backend's original still exists (ownership not transferred).
    backend_files = list(backend_root.glob("layer_0001_residual_post.*"))
    assert any(p.suffix == ".bin" for p in backend_files)
    assert any(p.suffix == ".json" for p in backend_files)

    # Hardlinked data file shares an inode with the backend file (cheap).
    backend_bin = backend_root / "layer_0001_residual_post.bin"
    if backend_bin.stat().st_dev == art.data_path.stat().st_dev:
        # Same filesystem — hardlink expected.
        assert backend_bin.stat().st_ino == art.data_path.stat().st_ino

    # Either way, contents match.
    via_path = _read_dense_memmap(art.data_path, art.sidecar_path)
    via_tensor = result.activations(layer=1, hook="residual_post")
    assert torch.equal(via_path, via_tensor)


@pytest.mark.integration
def test_as_path_ragged_inplace(tmp_path: Path) -> None:
    """Ragged shards: ResultArtifact has layout='ragged', shape=None,
    sidecar carries per-sample offsets so the file is self-describing."""
    from fpwap import RaggedTensor, ResultArtifact, Sweep
    from fpwap.storage.memmap import MemmapBackend

    model = _tiny_gpt2()
    keep = _per_sample_keep_lengths(N_SAMPLES, SEQ_LEN)
    backend_root = tmp_path / "backend"
    backend = MemmapBackend(root=backend_root)
    result = Sweep(
        model=model,
        dataset=_dataset(),
        seq_len=SEQ_LEN,
        callbacks=[_RaggedKeepFromTail(keep)],
        transport_dtype=torch.float32,
        microbatch_size=2,
        seed=SEED,
        progress=False,
        storage=backend,
    ).run()

    art = result.activations(layer=0, hook="residual_post", as_path=True)
    assert isinstance(art, ResultArtifact)
    assert art.layout == "ragged"
    assert art.shape is None
    assert art.data_path.exists()
    assert art.sidecar_path is not None and art.sidecar_path.exists()

    meta = json.loads(art.sidecar_path.read_text())
    assert meta["layout"] == "ragged"
    assert meta["offsets"][-1] == sum(keep)
    assert meta["offsets"][0] == 0
    assert len(meta["offsets"]) == N_SAMPLES + 1

    # The in-place read still works through the normal accessor.
    rt = result.activations(layer=0, hook="residual_post")
    assert isinstance(rt, RaggedTensor)
    assert int(rt.offsets[-1].item()) == sum(keep)

    # Self-describing: third party reconstructs the same RaggedTensor
    # from (data_path, sidecar_path) alone — symmetric with the dense test.
    via_path = _read_ragged_memmap(art.data_path, art.sidecar_path)
    assert torch.equal(via_path.flat, rt.flat)
    assert torch.equal(via_path.offsets, rt.offsets)


@pytest.mark.integration
def test_as_path_ragged_hardlink_to_dest(tmp_path: Path) -> None:
    """Dest mode for ragged: both data and sidecar (with offsets) move
    together so the destination is self-describing."""
    from fpwap import ResultArtifact, Sweep
    from fpwap.storage.memmap import MemmapBackend

    model = _tiny_gpt2()
    keep = _per_sample_keep_lengths(N_SAMPLES, SEQ_LEN)
    backend_root = tmp_path / "backend"
    dest = tmp_path / "user_dest"
    dest.mkdir()
    backend = MemmapBackend(root=backend_root)
    result = Sweep(
        model=model,
        dataset=_dataset(),
        seq_len=SEQ_LEN,
        callbacks=[_RaggedKeepFromTail(keep)],
        transport_dtype=torch.float32,
        microbatch_size=2,
        seed=SEED,
        progress=False,
        storage=backend,
    ).run()

    art = result.activations(layer=1, hook="residual_post", as_path=dest)
    assert isinstance(art, ResultArtifact)
    assert art.data_path.parent == dest
    assert art.sidecar_path is not None and art.sidecar_path.parent == dest
    meta = json.loads(art.sidecar_path.read_text())
    assert meta["layout"] == "ragged"
    assert meta["offsets"][-1] == sum(keep)

    # Self-describing round-trip via the dest paths matches the live
    # ragged accessor — same coverage the dense-dest test gives.
    from fpwap import RaggedTensor

    rt = result.activations(layer=1, hook="residual_post")
    assert isinstance(rt, RaggedTensor)
    via_path = _read_ragged_memmap(art.data_path, art.sidecar_path)
    assert torch.equal(via_path.flat, rt.flat)
    assert torch.equal(via_path.offsets, rt.offsets)


@pytest.mark.integration
def test_as_path_dest_collision_raises(tmp_path: Path) -> None:
    """Two `path_for(dest=same_dir)` calls for the same shard collide on
    the destination filename. We surface that with FileExistsError instead
    of silently clobbering a previous handle the caller may still hold."""
    from fpwap import Sweep
    from fpwap.callbacks.common import RawActivations
    from fpwap.storage.memmap import MemmapBackend

    model = _tiny_gpt2()
    backend_root = tmp_path / "backend"
    dest = tmp_path / "user_dest"
    dest.mkdir()
    backend = MemmapBackend(root=backend_root)
    result = Sweep(
        model=model,
        dataset=_dataset(),
        seq_len=SEQ_LEN,
        callbacks=[
            RawActivations(layers="all", last_token_only=False, out_dtype=torch.float32)
        ],
        transport_dtype=torch.float32,
        microbatch_size=2,
        seed=SEED,
        progress=False,
        storage=backend,
    ).run()

    result.activations(layer=0, hook="residual_post", as_path=dest)
    with pytest.raises(FileExistsError):
        result.activations(layer=0, hook="residual_post", as_path=dest)


@pytest.mark.parametrize("empty", ["", Path("")])
def test_as_path_empty_raises(empty) -> None:
    """as_path='' is falsy and would silently fall through to the in-memory
    branch; as_path=Path('') is truthy and resolves to cwd → hardlinks
    silently into the working dir. Both are config bugs — fail loud."""
    from fpwap.engine import Result

    res = Result(sweep_id="x")
    with pytest.raises(ValueError, match="non-empty"):
        res.activations(layer=0, hook="residual_post", as_path=empty)


def test_as_path_without_storage_raises() -> None:
    """No backend → no on-disk file to point at. Make the failure mode
    explicit instead of returning a misleading in-memory tensor wrapped
    in a fake path."""
    from fpwap.engine import Result

    res = Result(sweep_id="x")
    with pytest.raises((RuntimeError, ValueError)):
        res.activations(layer=0, hook="residual_post", as_path=True)
