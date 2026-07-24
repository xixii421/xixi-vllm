"""Resource layer (v2 ``kv/`` role): KVManager + Sequence + Triton KV ops.

Naming stays under ``memory/`` for historical paths; APIs match v2's KVManager
contract (can_allocate / allocate / sync_prefix / deallocate).
"""

from .block_manager import BlockManager
from .kv_ops import store_kvcache
from .manager import KVManager
from .sequence import Sequence, SequenceStatus, slot_of, slots_for_range

__all__ = [
    "BlockManager",
    "KVManager",
    "Sequence",
    "SequenceStatus",
    "slot_of",
    "slots_for_range",
    "store_kvcache",
]
