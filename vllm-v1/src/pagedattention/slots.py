""" 地址映射 """
from __future__ import annotations
import torch


def _slot_of(block_table: list[int], pos: int, block_size: int) -> int:
    """ 由一个逻辑位置转换到一个全局物理slot地址 
    
    Args:
        block_table: 一个列表，列表中每个元素是一个整数，表示一个物理页面的地址。
        pos: 一个整数，表示一个逻辑位置。
        block_size: 一个整数，表示一个物理页面的大小。
    Returns:
       一个整数，表示一个全局物理slot地址。
    """

    block_idx = pos // block_size          # 落在逻辑第几页

    # 超出页表范围， 抛出异常
    if block_idx >= len(block_table):
        raise IndexError("slot index out of range")

    # 返回物理slot地址
    return block_table[block_idx] * block_size + (pos % block_size)


def _slots_for_range(
    block_table: list[int],
    start: int, end: int, block_size: int,
) -> list[int]:
    """ 把一段逻辑区间变成一串slot
    
    Args:
        block_table: KV 页表
        start/end: 逻辑区间
        block_size: 物理页面大小
    Returns:
       一个列表，列表中每个元素是一个整数，表示一个全局物理slot地址。
    """

    if end < start: # 逻辑区间无效
        raise ValueError("invalid range")
    
    # 遍历逻辑区间，转换为物理slot地址
    return [_slot_of(block_table, pos, block_size) for pos in range(start, end)]


def slots_tensor(
    block_table: list[int],
    start: int, end: int, block_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """ 
    把一段逻辑区间变成一串slot，然后转换为Tensor

    Args:
        block_table: KV 页表
        start/end: 逻辑区间
        block_size: 物理页面大小
        device: 设备
    Returns:
       一个Tensor，表示一个全局物理slot地址。
    """
    # 把一段逻辑区间变成一串slot
    slots = _slots_for_range(block_table, start, end, block_size)
    # 转换为Tensor
    return torch.tensor(slots, device=device, dtype=torch.long)
