from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from torch import Tensor

from fpwap.types import ArtifactKey, HookName, fpwapArtifact


@dataclass
class ActivationSchema:
    shape: Sequence[int]
    dtype: str


@dataclass
class ShardHandle:
    path: str
    schema: ActivationSchema


@dataclass
class ShardManifest:
    path: str
    n_rows: int
    schema: ActivationSchema


class StorageBackend(Protocol):
    def open_shard(
        self,
        fpwap_id: str,
        layer_idx: int,
        hook: HookName,
        kind: str,
        schema: ActivationSchema,
    ) -> ShardHandle: ...

    def write_rows(
        self,
        handle: ShardHandle,
        rows: Tensor,
        sample_ids: Tensor,
    ) -> None: ...

    def close_shard(self, handle: ShardHandle) -> ShardManifest: ...

    def write_artifact(
        self,
        fpwap_id: str,
        key: ArtifactKey,
        artifact: fpwapArtifact,
    ) -> None: ...


__all__ = [
    "ActivationSchema",
    "ShardHandle",
    "ShardManifest",
    "StorageBackend",
]
