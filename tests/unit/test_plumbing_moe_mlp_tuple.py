"""Decomposed-path guard for MoE sparse blocks whose mlp returns a tuple.

On transformers < 5.6 the MoE sparse block returned
`(hidden_states, router_logits)`; 5.6 returns a plain tensor. The decomposed
`mlp_out` path in LlamaPlumbing does `residual + b.mlp(h)`, which breaks on
the tuple form. The plumbing must unwrap before the residual add so the
decomposed path is version-robust (issue #77, related item).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from fpwap.models.llama import LlamaPlumbing

HIDDEN = 8


class _TupleMlp(nn.Module):
    """Mimics a pre-5.6 MoE sparse block: returns (hidden, router_logits)."""

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        return h * 2.0, torch.zeros(h.shape[0], 4)


class _Attn(nn.Module):
    def forward(self, hidden_states: Tensor, **kwargs: object) -> tuple[Tensor, None]:
        return hidden_states * 0.5, None


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()
        self.self_attn = _Attn()
        self.mlp = _TupleMlp()


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block()])
        self.embed_tokens = nn.Embedding(4, HIDDEN)
        self.rotary_emb = lambda hidden_states, position_ids: (None, None)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Inner()


def test_decomposed_mlp_out_unwraps_tuple_returns() -> None:
    model = _Model()
    plumbing = LlamaPlumbing()
    block = model.model.layers[0]
    hidden = torch.randn(2, 3, HIDDEN)

    out, extras = plumbing.layer_forward_with_hooks(
        model,
        block,
        hidden,
        wanted_hooks=frozenset({"mlp_out"}),
    )

    # attn: h + 0.5h = 1.5h; mlp: 1.5h + 2*(1.5h) = 4.5h
    assert isinstance(out, Tensor)
    assert torch.allclose(out, hidden * 4.5)
    assert isinstance(extras["mlp_out"], Tensor)
    assert torch.allclose(extras["mlp_out"], hidden * 3.0)
