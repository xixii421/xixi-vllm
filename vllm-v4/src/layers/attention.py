import contextvars
from contextlib import contextmanager
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..memory.kv_ops import store_kvcache

# 使用 ContextVar 确保多线程/多协程并发安全
_ATTN_META: contextvars.ContextVar[Optional["AttentionMetadata"]] = contextvars.ContextVar("_ATTN_META", default=None)


@dataclass
class AttentionMetadata:
    wrapper: object
    kv_cache_pool: torch.Tensor
    slot_mapping: torch.Tensor
    output_kv_indices: Optional[torch.Tensor] = None
    # 预计算好 k_cache 和 v_cache 的 view，避免 forward 中重复 reshape
    k_cache_view: Optional[torch.Tensor] = None
    v_cache_view: Optional[torch.Tensor] = None


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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        meta = _get_metadata()
        
        # 写入 KV Cache
        store_kvcache(
            k,
            v,
            meta.k_cache_view,
            meta.v_cache_view,
            meta.slot_mapping,
        )

        # q.contiguous() 可能会触发内存拷贝。
        q_3d = q.contiguous().view(-1, self.num_heads, self.head_dim)
        
        # 调用 FlashInfer wrapper
        o = meta.wrapper.run(q_3d, (meta.k_cache_view, meta.v_cache_view))
        
        return o.view(-1, self.num_heads * self.head_dim)