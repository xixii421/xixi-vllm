""" KV 管理：块分配 + 前缀缓存；物理读写见 ``pagedattention``。

对外（包 ``__all__``）：
  Sequence / KVManager / ForwardContext
  set_forward_context / get_forward_context

常用跨文件公开（非 __all__，仍可 import）：
  BlockPool / HashPrefixIndex / RadixPrefixIndex / NoopPrefixIndex / compute_block_hash / RadixTree

命名约定：仅本文件使用 → ``_`` 前缀；跨文件引用 → 正常名。

与 pagedattention 分工：
  kv            — 谁占哪些页、前缀能否复用
  pagedattention — 给定页表如何 write / gather
"""

from .context import ForwardContext, get_forward_context, set_forward_context
from .manager import KVManager
from .sequence import Sequence

__all__ = [
    "ForwardContext",
    "KVManager",
    "Sequence",
    "get_forward_context",
    "set_forward_context",
]
