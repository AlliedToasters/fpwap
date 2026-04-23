from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

from fpwap.callbacks.base import Callback
from fpwap.types import (
    Artifact,
    BatchResult,
    HookName,
    LayerArtifact,
)


class RawActivations(Callback):
    """Persist per-sample activations, pooled (`last_token_only=True`) by default.

    Returns an `Emit` each microbatch; the engine routes these into the run's
    result so `result.activations(layer, hook)` can return them concatenated
    in sample order. For datasets too large to hold in memory, swap in a
    disk-backed StorageBackend.
    """

    phase = "read"

    def __init__(
        self,
        layers: Sequence[int] | Literal["all"] = "all",
        hook: HookName = "residual_post",
        last_token_only: bool = True,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.target_layers = layers
        self.target_hooks = (hook,)
        self.last_token_only = last_token_only
        self.out_dtype = out_dtype

    def on_batch(
        self,
        layer_idx: int,
        hook: HookName,
        acts: Tensor,
        sample_ids: Tensor,
    ) -> BatchResult:
        from fpwap.types import Emit

        pooled = acts[:, -1, :] if self.last_token_only else acts
        return Emit(pooled.to(self.out_dtype), dtype=self.out_dtype)


class IncrementalPCA(Callback):
    """Streaming PCA over the dataset. One pass, O(H²) memory per layer.

    Accumulates the running mean and `X^T X` in fp32 across microbatches;
    at `on_layer_end` computes `cov = E[XX^T] - mean mean^T` and returns
    the top-k eigenvectors as a `LayerArtifact(kind="pca_basis", ...)`.
    The engine routes it into `result.artifact("pca_basis", layer=i)`.

    Pooling: `last_token_only=True` by default (matches RawActivations).
    Any 3D `[N, S, H]` activation is pooled to `[N, H]` before accumulation;
    users needing other pooling should pre-pool in an upstream callback or
    subclass. Accumulators run on the execution device to avoid per-batch
    H2D; the final basis is moved to CPU in the artifact payload.
    """

    phase = "read"

    def __init__(
        self,
        layers: Sequence[int] | Literal["all"] = "all",
        n_components: int = 64,
        hook: HookName = "residual_post",
        last_token_only: bool = True,
    ) -> None:
        self.target_layers = layers
        self.target_hooks = (hook,)
        self.n_components = n_components
        self.last_token_only = last_token_only
        self._sums: dict[int, Tensor] = {}
        self._sumxx: dict[int, Tensor] = {}
        self._counts: dict[int, int] = {}

    def on_batch(
        self,
        layer_idx: int,
        hook: HookName,
        acts: Tensor,
        sample_ids: Tensor,
    ) -> BatchResult:
        x = acts[:, -1, :] if (self.last_token_only and acts.dim() == 3) else acts
        if x.dim() != 2:
            raise ValueError(
                f"IncrementalPCA expects 2D activations [N, H] after pooling, "
                f"got {tuple(x.shape)}"
            )
        x = x.to(torch.float32)
        hdim = x.shape[-1]
        if layer_idx not in self._sums:
            self._sums[layer_idx] = torch.zeros(hdim, dtype=torch.float32, device=x.device)
            self._sumxx[layer_idx] = torch.zeros(
                hdim, hdim, dtype=torch.float32, device=x.device
            )
            self._counts[layer_idx] = 0
        self._sums[layer_idx] += x.sum(dim=0)
        self._sumxx[layer_idx] += x.T @ x
        self._counts[layer_idx] += int(x.shape[0])
        return None

    def on_layer_end(self, layer_idx: int) -> LayerArtifact | None:
        if layer_idx not in self._sums:
            return None
        n = self._counts[layer_idx]
        if n == 0:
            return None
        mean = self._sums[layer_idx] / n
        cov = self._sumxx[layer_idx] / n - torch.outer(mean, mean)
        # eigh returns ascending eigenvalues; reverse for PCA convention.
        eigvals, eigvecs = torch.linalg.eigh(cov)
        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        k = min(self.n_components, eigvecs.shape[1])
        basis = eigvecs[:, :k].contiguous()
        # Drop per-layer accumulators once artifact is materialized.
        del self._sums[layer_idx]
        del self._sumxx[layer_idx]
        del self._counts[layer_idx]
        return LayerArtifact(
            kind="pca_basis",
            payload={
                "basis": basis.cpu(),
                "mean": mean.cpu(),
                "explained_variance": eigvals[:k].cpu(),
            },
        )


class DiffOfMeans(Callback):
    """Per-class activation means for binary-labeled data.

    User-supplied `labels` is a 1D int tensor indexed by sample_id — this
    sidesteps the dataset-items-in-callback API question. Accumulates
    streaming sums per class (in fp32, on the execution device), and at
    `on_layer_end` returns a `LayerArtifact(kind="diff_of_means", payload)`
    carrying `(mean_1 - mean_0)`, both class means, and their counts. The
    returned direction is a common probing target ("does the model encode
    label L in this layer's residual stream?").

    Pooling: `last_token_only=True` by default, matching RawActivations.
    """

    phase = "read"

    def __init__(
        self,
        labels: Tensor,
        layers: Sequence[int] | Literal["all"] = "all",
        hook: HookName = "residual_post",
        last_token_only: bool = True,
    ) -> None:
        self.target_layers = layers
        self.target_hooks = (hook,)
        self.labels = labels.to(torch.int64)
        self.last_token_only = last_token_only
        self._sums: dict[int, dict[int, Tensor]] = {}
        self._counts: dict[int, dict[int, int]] = {}

    def on_batch(
        self,
        layer_idx: int,
        hook: HookName,
        acts: Tensor,
        sample_ids: Tensor,
    ) -> BatchResult:
        x = acts[:, -1, :] if (self.last_token_only and acts.dim() == 3) else acts
        if x.dim() != 2:
            raise ValueError(
                f"DiffOfMeans expects 2D activations [N, H] after pooling, "
                f"got {tuple(x.shape)}"
            )
        x = x.to(torch.float32)
        # sample_ids → labels (cpu lookup into user-provided tensor, then push
        # back to acts' device for the masked-sum below).
        labels = self.labels[sample_ids.detach().cpu()].to(x.device)
        sums = self._sums.setdefault(layer_idx, {})
        counts = self._counts.setdefault(layer_idx, {})
        for cls in torch.unique(labels).tolist():
            mask = labels == cls
            cls_x = x[mask]
            if cls not in sums:
                sums[cls] = torch.zeros(x.shape[-1], dtype=torch.float32, device=x.device)
                counts[cls] = 0
            sums[cls] += cls_x.sum(dim=0)
            counts[cls] += int(cls_x.shape[0])
        return None

    def on_layer_end(self, layer_idx: int) -> LayerArtifact | None:
        if layer_idx not in self._sums:
            return None
        sums = self._sums[layer_idx]
        counts = self._counts[layer_idx]
        means = {cls: sums[cls] / counts[cls] for cls in sums}
        # Drop accumulators.
        del self._sums[layer_idx]
        del self._counts[layer_idx]
        # Binary-labeled convention: require at least labels 0 and 1 present.
        if 0 in means and 1 in means:
            direction = means[1] - means[0]
        else:
            direction = None
        return LayerArtifact(
            kind="diff_of_means",
            payload={
                "direction": direction.cpu() if direction is not None else None,
                "means": {int(c): v.cpu() for c, v in means.items()},
                "counts": {int(c): counts[c] for c in counts},
            },
        )


class SteerInBasis(Callback):
    """Additive intervention in a pre-computed basis.

    `acts + alpha * basis[:, direction_idx]`, broadcast across the batch and
    sequence dims. `basis_artifact.payload["basis"]` is expected to be
    `[H, n_components]` (the shape produced by `IncrementalPCA`); any
    Artifact with a compatible payload works. Lives in `phase="write"`, so
    the returned `WriteBack` replaces the residual that feeds the next
    layer (or the buffer, if targeted at `residual_post`).
    """

    phase = "write"

    def __init__(
        self,
        basis_artifact: Artifact,
        direction_idx: int,
        alpha: float,
        layers: Sequence[int] | Literal["all"] = "all",
        hook: HookName = "residual_post",
    ) -> None:
        self.target_layers = layers
        self.target_hooks = (hook,)
        self.basis = basis_artifact
        self.direction_idx = direction_idx
        self.alpha = alpha

    def on_batch(
        self,
        layer_idx: int,
        hook: HookName,
        acts: Tensor,
        sample_ids: Tensor,
    ) -> BatchResult:
        from fpwap.types import WriteBack

        payload = self.basis.payload
        if isinstance(payload, dict):
            # Prefer "basis" [H, K]; fall back to "direction" [H] (what
            # DiffOfMeans returns). direction_idx is ignored for 1D payloads.
            if "basis" in payload:
                basis = payload["basis"]
                direction = basis[:, self.direction_idx]
            elif "direction" in payload and payload["direction"] is not None:
                direction = payload["direction"]
            else:
                raise KeyError(
                    "basis_artifact payload has no 'basis' or 'direction' key"
                )
        else:
            direction = (
                payload[:, self.direction_idx] if payload.dim() == 2 else payload
            )
        direction = direction.to(device=acts.device, dtype=acts.dtype)
        return WriteBack(acts + self.alpha * direction)
