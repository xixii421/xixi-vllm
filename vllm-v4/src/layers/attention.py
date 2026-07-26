import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..memory.kv_ops import store_kvcache

_ATTN_META: contextvars.ContextVar[Optional["AttentionMetadata"]] = contextvars.ContextVar(
    "_ATTN_META", default=None
)


@dataclass
class AttentionMetadata:
    wrapper: object
    kv_cache_pool: torch.Tensor
    slot_mapping: torch.Tensor
    output_kv_indices: Optional[torch.Tensor] = None


@contextmanager
def attention_context(meta: AttentionMetadata):
    token = _ATTN_META.set(meta)
    try:
        yield
    finally:
        _ATTN_META.reset(token)


def _get_metadata() -> AttentionMetadata:
    meta = _ATTN_META.get()
    assert meta is not None, "attention_context() not active"
    return meta


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        layer_id: int = 0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.layer_id = layer_id

    @torch.compiler.disable
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        meta = _get_metadata()
        pool = meta.kv_cache_pool
        n_kv = self.num_kv_heads
        hd = self.head_dim
        lid = self.layer_id

        # Per-layer paged KV (NHD). Must NOT come from a shared meta view —
        # every decoder layer owns pool[:, layer_id].
        k_4d = pool[0, lid]
        v_4d = pool[1, lid]
        store_kvcache(
            k,
            v,
            k_4d.reshape(-1, n_kv, hd),
            v_4d.reshape(-1, n_kv, hd),
            meta.slot_mapping,
        )

        q_3d = q.view(-1, self.num_heads, hd)
        if not q_3d.is_contiguous():
            q_3d = q_3d.contiguous()
        o = meta.wrapper.run(q_3d, (k_4d, v_4d))
        return o.view(-1, self.num_heads * hd)
