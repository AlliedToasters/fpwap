from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from torch import Tensor, nn

DispatchFn = Callable[[str, Tensor], Tensor]


class ModelPlumbing(Protocol):
    """Per-family hook plumbing. Implementations live in fpwap/models/<family>.py."""

    def matches(self, model: nn.Module) -> bool: ...

    def layer_modules(self, model: nn.Module) -> Sequence[nn.Module]: ...

    def layer_prefix(self, layer_idx: int) -> str:
        """Absolute dotted path to the i-th transformer block.

        Used to translate a block's relative parameter names into the fully
        qualified names that live in the accelerate index (e.g. GPT-2's
        "transformer.h.5", Llama's "model.layers.5").
        """
        ...

    def embedding_param_names(self, model: nn.Module) -> Sequence[str]:
        """Absolute names of parameters used by embed() — loaded once and
        kept resident on the execution device (SPEC D.5). Excludes block
        params (those stream per-layer).
        """
        ...

    def embed(self, model: nn.Module, input_ids: Tensor) -> Tensor: ...

    def layer_forward_with_hooks(
        self,
        model: nn.Module,
        block: nn.Module,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        wanted_hooks: frozenset[str] = frozenset(),
        dispatch_fn: DispatchFn | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Run a single transformer block, optionally exposing sub-layer outputs.

        Returns `(residual_out, extras)` where:
          - `residual_out` is the block's full output (same as `residual_post`).
          - `extras` carries requested sub-layer outputs (post-dispatch):
              * "attn_out" — attention sub-layer output, BEFORE the residual add
              * "mlp_out"  — MLP sub-layer output, BEFORE the residual add

        If `dispatch_fn` is provided, the plumbing calls it at each sub-layer
        boundary for hooks in `wanted_hooks`; its return value replaces the
        tensor flowing into the subsequent residual add. This enables
        WriteBack at sub-layer hooks (zero-out attention, scale MLP, etc.).
        When `dispatch_fn` is None, the extras are captured as-is and the
        engine dispatches them read-only after the fact.

        `residual_pre` is the caller's `hidden_states` argument, so it is
        dispatched at the engine level and not returned here.

        When `wanted_hooks` is empty (no callback targets `attn_out`/`mlp_out`),
        implementations MUST take the fast path — calling `block(...)` directly
        — to avoid the per-microbatch overhead of decomposing the block.
        """
        ...
