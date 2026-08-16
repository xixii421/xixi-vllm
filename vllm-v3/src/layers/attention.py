"""Attention 执行层。
该模块定义了 attention 的调用契约、执行上下文和唯一执行入口。
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import NamedTuple, Optional
import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func
from ..memory.kv_ops import store_kvcache


class AttnInputs(NamedTuple):
    """每次 attention 调用时随 batch 变化的输入。

    与 AttentionMetadata 的分工：
        - AttnInputs：每步都变化、需要作为图输入显式传递的参数。
        - AttentionMetadata：跨层共享、不进图的执行上下文，通过上下文管理器传递。
    """

    # [total_tokens] int32，每个 token 写入 KV cache 的物理槽位；-1 表示 padding，跳过写入。
    slot_mapping: torch.Tensor

    # [n_seq + 1] int32，序列累计边界，仅 FA2 varlen 路径使用；decode / paged 路径忽略。
    cu_seqlens: torch.Tensor

    # 本批最大序列长度，仅 FA2 varlen 路径使用；decode / paged 路径忽略。
    max_seqlen: int

    # True 表示首 chunk prefill，使用 FA2 varlen；False 表示 decode 或续 chunk，使用 FlashInfer paged。
    use_fa2: bool


@dataclass
class AttentionMetadata:
    """attention 执行所需的跨层共享上下文。

    该对象通过 attention_context 上下文管理器在调用栈中传递。

    Attributes:
        kv_cache_pool: engine 级 KV cache 物理池。
        wrapper: 当前 batch 的 FlashInfer 分页 wrapper 实例
    """

    kv_cache_pool: torch.Tensor
    wrapper: object = None


# 当前激活的 attention 执行上下文，由 attention_context 管理。
_ATTN_META: Optional["AttentionMetadata"] = None


@contextmanager
def attention_context(meta: AttentionMetadata):
    """设置当前线程的 attention 执行上下文。

    在 with 块内，attention 函数可通过 _ATTN_META 访问
    kv_cache_pool 和 wrapper。支持嵌套调用，退出时自动恢复上一层上下文。

    Args:
        meta: 本次调用使用的 AttentionMetadata。

    Yields:
        None
    """
    global _ATTN_META
    prev = _ATTN_META
    _ATTN_META = meta
    try:
        yield
    finally:
        _ATTN_META = prev


@torch.library.custom_op("vllm::attention", mutates_args=())
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slot_mapping: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    use_fa2: bool,
    layer_id: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """执行 attention 前向。"""
    meta = _ATTN_META
    assert meta is not None, "attention_context() not active"
    pool = meta.kv_cache_pool

    # 将本步 K、V 写入 KV cache 对应层的物理槽位。
    store_kvcache(
        k,
        v,
        pool[0, layer_id].reshape(-1, num_kv_heads, head_dim),
        pool[1, layer_id].reshape(-1, num_kv_heads, head_dim),
        slot_mapping,
    )

    q = q.contiguous().view(-1, num_qo_heads, head_dim)

    if use_fa2:
        # 首 chunk prefill：使用 flash-attn varlen 接口，支持变长序列 causal attention。
        o = flash_attn_varlen_func(
            q,
            k.contiguous(),
            v.contiguous(),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
        )
    else:
        # decode 或续 chunk prefill：使用 FlashInfer 分页 wrapper 执行 paged attention。
        o = meta.wrapper.run(q, (pool[0, layer_id], pool[1, layer_id]))

    return o.view(-1, num_qo_heads * head_dim)


@attention.register_fake
def _attention_fake(
    q, k, v, slot_mapping, cu_seqlens, max_seqlen, use_fa2,
    layer_id, num_qo_heads, num_kv_heads, head_dim,
):
    """attention 的 fake 实现，供 torch.compile 形状推导使用。"""
    return torch.empty(
        (q.shape[0], num_qo_heads * head_dim), dtype=q.dtype, device=q.device
    )


class Attention(nn.Module):
    """Transformer 层中的 attention 模块。

    将 q、k、v 和批次元数据封装后，委托给 attention custom op 执行。
    每个 transformer 层持有一个 Attention 实例，layer_id 用于定位 KV cache 中对应的层。
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        layer_id: int = 0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.layer_id = layer_id

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn: AttnInputs) -> torch.Tensor:
        """执行 attention 前向。

        Args:
            q: query 张量。
            k: key 张量。
            v: value 张量。
            attn: 本步随 batch 变化的 attention 输入元数据。

        Returns:
            attention 输出张量。
        """
        return torch.ops.vllm.attention(
            q, k, v,
            attn.slot_mapping, attn.cu_seqlens, attn.max_seqlen, attn.use_fa2,
            self.layer_id, self.num_heads, self.num_kv_heads, self.head_dim,
        )