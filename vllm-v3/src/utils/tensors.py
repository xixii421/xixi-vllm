"""CPU → GPU 张量传输工具。"""
from __future__ import annotations

import torch


def h2d(
    data: list[int] | list[list[int]],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """CPU 列表 → GPU 张量实现pin_memory + non_blocking 异步拷贝"""
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)
