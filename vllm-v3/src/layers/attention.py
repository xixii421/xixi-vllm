from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..memory.kv_ops import store_kvcache
from ..utils.context import get_context

_ATTN_META: Optional["AttentionMetadata"] = None


@dataclass
class AttentionMetadata:
    wrapper: object
    kv_cache_pool: torch.Tensor
    slot_mapping: torch.Tensor
    output_kv_indices: Optional[torch.Tensor] = None
    # 首 chunk prefill 用 FlashAttention-2（比 FlashInfer prefill 快 2-3x）。
    # decode / 续 chunk / prefill CG 仍走 wrapper（FlashInfer paged）。
    use_fa2: bool = False


@contextmanager
def attention_context(meta: AttentionMetadata):
    global _ATTN_META
    prev = _ATTN_META
    _ATTN_META = meta
    try:
        yield
    finally:
        _ATTN_META = prev


def _get_metadata() -> AttentionMetadata:
    assert _ATTN_META is not None, "attention_context() not active"
    return _ATTN_META


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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        meta = _get_metadata()
        pool = meta.kv_cache_pool
        n_kv = self.num_kv_heads
        hd = self.head_dim
        lid = self.layer_id

        k_4d = pool[0, lid]
        v_4d = pool[1, lid]

        k_4d_2d = k_4d.reshape(-1, n_kv, hd)
        v_4d_2d = v_4d.reshape(-1, n_kv, hd)
        store_kvcache(k, v, k_4d_2d, v_4d_2d, meta.slot_mapping)

        q = q.contiguous().view(-1, self.num_heads, hd)

        if meta.use_fa2:
            from flash_attn import flash_attn_func, flash_attn_varlen_func

            ctx = get_context()
            # 多 seq 首 chunk：必须用 varlen 让每条 seq 各自 causal。
            if ctx.cu_seqlens_q is not None and ctx.cu_seqlens_q.numel() > 2:
                o = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q=ctx.cu_seqlens_q,
                    cu_seqlens_k=ctx.cu_seqlens_k,
                    max_seqlen_q=ctx.max_seqlen_q,
                    max_seqlen_k=ctx.max_seqlen_k,
                    causal=True,
                )
            else:
                o = flash_attn_func(
                    q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), causal=True
                )[0]
            return o.view(-1, self.num_heads * hd)

        o = meta.wrapper.run(q, (k_4d, v_4d))
        return o.view(-1, self.num_heads * hd)
