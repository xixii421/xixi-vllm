"""Per-forward metadata for KV Cache (prefill vs decode)."""

from __future__ import annotations

from dataclasses import dataclass

from ..kv_cache import KVCache


@dataclass
class ForwardContext:
    """Describes one model forward pass.

    cache_len is the number of K/V slots already filled before this step writes.
    Prefill writes new tokens starting at cache_len.
    Decode writes exactly one new token at slot cache_len.
    """

    is_prefill: bool
    cache_len: int
    kv_cache: KVCache


_forward_ctx: ForwardContext | None = None


def set_forward_context(ctx: ForwardContext | None) -> None:
    global _forward_ctx
    _forward_ctx = ctx


def get_forward_context() -> ForwardContext | None:
    return _forward_ctx
