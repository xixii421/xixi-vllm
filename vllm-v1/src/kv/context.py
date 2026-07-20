""" 单次 forward 的页表合同（Engine → Attention） """

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..pagedattention import PagedStore


@dataclass
class ForwardContext:
    """ Attention 只用页表合同。 """

    is_prefill: bool              # 是否 prefill 步
    cache_len: int                # 本步开始前已有 KV 长度
    seq_len: int                  # 本步写完后的总 KV 长度（= gather 的长度）
    slot_mapping: torch.Tensor    # 本步要写入的物理 slot
    block_table: list[int]        # 逻辑页 → 物理块号
    store: PagedStore             # 物理读写模块


_forward_ctx: ForwardContext | None = None


def set_forward_context(ctx: ForwardContext | None) -> None:
    """ 安装或清空当前线程的 ForwardContext。 """
    global _forward_ctx
    _forward_ctx = ctx


def get_forward_context() -> ForwardContext | None:
    """ 读取当前 ForwardContext；无则 Attention 走无缓存 SDPA。 """
    return _forward_ctx
