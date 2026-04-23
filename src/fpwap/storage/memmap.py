from __future__ import annotations

from pathlib import Path

from torch import Tensor

from fpwap.storage import ActivationSchema, ShardHandle, ShardManifest
from fpwap.types import ArtifactKey, HookName, fpwapArtifact


class MemmapBackend:
    """Default storage backend: memmap shards + parquet index."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def open_shard(
        self,
        fpwap_id: str,
        layer_idx: int,
        hook: HookName,
        kind: str,
        schema: ActivationSchema,
    ) -> ShardHandle:
        raise NotImplementedError

    def write_rows(
        self,
        handle: ShardHandle,
        rows: Tensor,
        sample_ids: Tensor,
    ) -> None:
        raise NotImplementedError

    def close_shard(self, handle: ShardHandle) -> ShardManifest:
        raise NotImplementedError

    def write_artifact(
        self,
        fpwap_id: str,
        key: ArtifactKey,
        artifact: fpwapArtifact,
    ) -> None:
        raise NotImplementedError
