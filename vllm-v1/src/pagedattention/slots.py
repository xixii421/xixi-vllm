""" 逻辑 token 区间 → GPU 物理 slot 地址映射 """

from __future__ import annotations

import torch


def slots_tensor(
    block_table: list[int],
    start: int,
    end: int,
    block_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """ 把逻辑区间 [start, end) 内的每个位置换算成全局物理 slot。

    slot = 所在物理页号 * 页大小 + 页内偏移；页号由 block_table[pos // block_size] 给出。
    """
    if end < start:
        raise ValueError("invalid range")
    slots = [
        block_table[pos // block_size] * block_size + (pos % block_size)
        for pos in range(start, end)
    ]
    return torch.tensor(slots, device=device, dtype=torch.long)
