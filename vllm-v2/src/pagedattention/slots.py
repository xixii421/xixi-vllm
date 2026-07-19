from __future__ import annotations
import torch


def slot_of(block_table: list[int], pos: int, block_size: int) -> int:
    block_idx = pos // block_size
    if block_idx >= len(block_table):
        raise IndexError(
            f"pos={pos} needs block_idx={block_idx}, "
            f"but block_table has {len(block_table)} pages"
        )
    return block_table[block_idx] * block_size + (pos % block_size)


def slots_for_range(
    block_table: list[int],
    start: int,
    end: int,
    block_size: int,
) -> list[int]:
    if end < start:
        raise ValueError(f"end={end} < start={start}")
    return [slot_of(block_table, pos, block_size) for pos in range(start, end)]


def slots_tensor(
    block_table: list[int],
    start: int,
    end: int,
    block_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    slots = slots_for_range(block_table, start, end, block_size)
    return torch.tensor(slots, device=device, dtype=torch.long)
