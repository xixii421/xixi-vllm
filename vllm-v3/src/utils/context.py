from __future__ import annotations
from dataclasses import dataclass, field

import torch
from ..memory import PagedKVCache


@dataclass
class ForwardContext:
    is_prefill: bool
    kv_cache: PagedKVCache
    block_tables: list[list[int]]
    slot_mapping: torch.Tensor
    query_lens: list[int]
    context_lens: list[int]
    cached_lens: list[int] = field(default_factory=list)


_forward_ctx: ForwardContext | None = None


def set_forward_context(ctx: ForwardContext | None) -> None:
    global _forward_ctx
    _forward_ctx = ctx


def get_forward_context() -> ForwardContext | None:
    return _forward_ctx
