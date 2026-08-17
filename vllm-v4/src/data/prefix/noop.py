"""No-op prefix index."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..block_manager import BlockManager
    from ..sequence import Sequence


class NoopPrefixIndex:
    def match(self, seq: Sequence, pool: BlockManager) -> list[int]:
        return []

    def publish(
        self,
        seq: Sequence,
        num_full_blocks: int,
        pool: BlockManager,
        *,
        start: int = 0,
    ) -> None:
        del seq, num_full_blocks, pool, start
        return

    def invalidate(self, block_id: int, pool: BlockManager) -> None:
        return

    def release_seq(self, seq_id: int) -> None:
        return
