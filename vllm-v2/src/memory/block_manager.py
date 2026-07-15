from __future__ import annotations
from collections import deque


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_block_ids: deque[int] = deque(range(num_blocks))

    @property
    def num_free(self) -> int:
        return len(self.free_block_ids)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self.free_block_ids)

    def blocks_for_tokens(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0
        return (num_tokens + self.block_size - 1) // self.block_size

    def _alloc_one(self) -> int:
        if not self.free_block_ids:
            raise RuntimeError(
                f"KV block pool exhausted: used={self.num_used}/{self.num_blocks}"
            )
        return self.free_block_ids.popleft()

    def ensure_capacity(self, block_table: list[int], num_tokens: int) -> None:
        need = self.blocks_for_tokens(num_tokens)
        while len(block_table) < need:
            block_table.append(self._alloc_one())

    def allocate(self, num_tokens: int) -> list[int]:
        table: list[int] = []
        self.ensure_capacity(table, num_tokens)
        return table

    def may_append(self, block_table: list[int], num_tokens: int) -> None:

        self.ensure_capacity(block_table, num_tokens)

    def free(self, block_table: list[int]) -> None:
        for bid in reversed(block_table):
            self.free_block_ids.append(bid)
        block_table.clear()
