from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from ...data.kv_ops import store_kvcache


class AttnInputs(NamedTuple):
    slot_mapping: torch.Tensor
    is_prefill: bool
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    block_tables: Optional[torch.Tensor] = None
    cache_seqlens: Optional[torch.Tensor] = None


@dataclass
class AttnMetadata: # 存放注意力算子需要的全局不变信息
    kv_cache: torch.Tensor


_ATTN_META: Optional[AttnMetadata] = None # 进程全局变量


@contextmanager
def attention_context(meta: AttnMetadata):
    global _ATTN_META
    prev = _ATTN_META
    _ATTN_META = meta
    try:
        yield
    finally:
        _ATTN_META = prev # 恢复旧值


@torch.library.custom_op("vllm::attention", mutates_args=())
# 注册torch库自定义算子，mutates_args，声明本函数没有直接修改输入参数
# compile捕获计算图遇到if会graph break，无法生成完整的triton图
# 用custom_op包装后，compile把整个vllm:attention当作单个黑盒算子节点
# 代价是算子内部的python代码不会被compile优化，真正加速来自内部调用flash-attention CUDA内核
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slot_mapping: torch.Tensor,
    is_prefill: bool,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_tables: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    layer_id: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    meta = _ATTN_META # 全局线程上下文对象，不作参数传入，直接从全局取
    assert meta is not None, "attention_context() not active"
    pool = meta.kv_cache
    k_cache = pool[0, layer_id]
    v_cache = pool[1, layer_id]
    store_kvcache(
        k,
        v,
        k_cache.reshape(-1, num_kv_heads, head_dim),
        v_cache.reshape(-1, num_kv_heads, head_dim),
        slot_mapping,
    ) # CUDA 拷贝内核， 把本步生成的kv写入kvcache
    q = q.contiguous().view(-1, num_qo_heads, head_dim)
    if is_prefill:
        if block_tables is not None:
            k, v = k_cache, v_cache
        else:
            # 不采用 paged kv，直接用新生成kv计算
            k = k.contiguous().view(-1, num_kv_heads, head_dim)
            v = v.contiguous().view(-1, num_kv_heads, head_dim)
        o = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=True,
            block_table=block_tables,
        )
    else:
        # decode阶段
        o = flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_tables,
            causal=True,
        )
    return o.view(-1, num_qo_heads * head_dim) # 打回hidden-states格式


@attention.register_fake
# 给自定义算子提供一个纯 Python 的形状模拟实现，只算输出 shape/dtype/device，不跑真实 GPU 计算
def _attention_fake(
    q, k, v, slot_mapping, is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
    max_seqlen_k, block_tables, cache_seqlens, layer_id, num_qo_heads,
    num_kv_heads, head_dim,
):
    return torch.empty(
        (q.shape[0], num_qo_heads * head_dim), dtype=q.dtype, device=q.device
    )


class Attention(nn.Module):
    def __init__(
        self, num_heads: int, head_dim: int, num_kv_heads: int, layer_id: int = 0
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.layer_id = layer_id

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn: AttnInputs
    ) -> torch.Tensor:
        return torch.ops.vllm.attention(
            q,
            k,
            v,
            attn.slot_mapping,
            attn.is_prefill,
            attn.cu_seqlens_q,
            attn.cu_seqlens_k,
            attn.max_seqlen_q,
            attn.max_seqlen_k,
            attn.block_tables,
            attn.cache_seqlens,
            self.layer_id,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )
