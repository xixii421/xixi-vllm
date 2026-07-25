from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .kv_ops import store_kvcache

_ATTN_META: ContextVar[AttentionMetadata | None] = ContextVar(
    "attention_metadata", default=None
)


@dataclass(frozen=True, slots=True)
class AttentionMetadata:
    """Per-step attention state shared by every Attention layer."""

    wrapper: Any
    kv_cache: torch.Tensor
    slot_mapping: torch.Tensor


@contextmanager
def attention_context(meta: AttentionMetadata):
    token = _ATTN_META.set(meta)
    try:
        yield
    finally:
        _ATTN_META.reset(token)


def _meta() -> AttentionMetadata:
    meta = _ATTN_META.get()
    assert meta is not None, "call forward inside attention_context()"
    return meta


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        layer_id: int = 0,
    ) -> None:
        super().__init__()
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.layer_id = layer_id

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q: [T, H, D]   k/v: [T, H_kv, D]
        assert q.ndim == 3 and q.shape[1:] == (self.num_heads, self.head_dim)
        assert k.shape == v.shape == (q.shape[0], self.num_kv_heads, self.head_dim)

        meta = _meta()
        k_cache = meta.kv_cache[0, self.layer_id]  # [blocks, page, H_kv, D]
        v_cache = meta.kv_cache[1, self.layer_id]
        # store_kvcache accepts QKV-split strided K/V when heads are packed.
        store_kvcache(
            k,
            v,
            k_cache.view(-1, self.num_kv_heads, self.head_dim),
            v_cache.view(-1, self.num_kv_heads, self.head_dim),
            meta.slot_mapping,
        )

        # RoPE writes contiguous Q; only copy if a caller passed a strided view.
        q_run = q if q.is_contiguous() else q.contiguous()
        out = meta.wrapper.run(q_run, (k_cache, v_cache))
        assert out.shape == q.shape, f"attn out {out.shape} != q {q.shape}"
        return out.flatten(1, 2)  # [T, H*D]
