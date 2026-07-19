from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .manager import KVManager
    from .sequence import Sequence


@dataclass
class ForwardContext:
    is_prefill: bool
    cache_len: int
    manager: KVManager
    seq: Sequence


_forward_ctx: ForwardContext | None = None


def set_forward_context(ctx: ForwardContext | None) -> None:
    global _forward_ctx
    _forward_ctx = ctx


def get_forward_context() -> ForwardContext | None:
    return _forward_ctx
