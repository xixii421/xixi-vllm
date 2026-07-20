""" 关闭跨请求前缀共享 """

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pool import BlockPool
    from ..sequence import Sequence


class NoopPrefixIndex:
    """ match 恒为空；publish / invalidate 为空操作。 """

    def match(self, seq: Sequence, pool: BlockPool) -> list[int]:
        """ 永不命中。 """
        return []

    def publish(self, seq: Sequence, num_full_blocks: int, pool: BlockPool) -> None:
        """ 不发布。 """
        return

    def invalidate(self, block_id: int, pool: BlockPool) -> None:
        """ 无索引可删。 """
        return
