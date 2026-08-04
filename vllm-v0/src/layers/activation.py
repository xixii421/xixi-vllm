from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SiluAndMul(nn.Module):
    """silu and mul 组合算子"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x, y 前边是value, 后边是gate，对应 silu(gate) * value
        x, y = x.chunk(2, dim=-1)
        return F.silu(y) * x
