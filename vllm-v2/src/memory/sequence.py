from __future__ import annotations


def slot_of(block_table: list[int], pos: int, block_size: int) -> int:
    block_idx = pos // block_size
    if block_idx >= len(block_table):
        raise IndexError(
            f"pos={pos} needs block_idx={block_idx}, but block_table has {len(block_table)} pages"
        )
    return block_table[block_idx] * block_size + (pos % block_size)


def slots_for_range(
    block_table: list[int],
    start: int,
    end: int,
    block_size: int,
) -> list[int]:
    return [slot_of(block_table, pos, block_size) for pos in range(start, end)]


class Sequence:
    def __init__(self, token_ids: list[int] | None = None) -> None:
        self.token_ids: list[int] = list(token_ids or [])
        self.block_table: list[int] = []

    def __len__(self) -> int:
        return len(self.token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)
