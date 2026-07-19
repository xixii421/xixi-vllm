"""PagedAttention 教学模块：页表映射 + 按 slot 写 / 按 block_table 读。

与 ``src.kv`` 的分工：
- **本包**：纯存储契约（高内聚）——不感知前缀缓存 / Sequence 生命周期
- **kv**：块分配、前缀索引、Engine 编排（通过 ForwardContext 注入本包）

Attention 只依赖 ``PagedStore`` + ``slot_mapping`` / ``block_table``，与 KVManager 解耦。
"""

from .slots import slot_of, slots_for_range, slots_tensor
from .store import PagedStore

__all__ = [
    "PagedStore",
    "slot_of",
    "slots_for_range",
    "slots_tensor",
]
