"""Per-forward metadata for PagedAttention (prefill vs decode)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..memory import PagedKVCache


@dataclass
class ForwardContext:
    is_prefill: bool
    seq_len: int
    kv_cache: PagedKVCache
    block_table: list[int]
    slot_mapping: torch.Tensor


_forward_ctx: ForwardContext | None = None


def set_forward_context(ctx: ForwardContext | None) -> None:
    global _forward_ctx
    _forward_ctx = ctx


def get_forward_context() -> ForwardContext | None:
    return _forward_ctx
