"""SwiGLU 用的门控激活：SiLU(gate) * up。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SiluAndMul(nn.Module):
    """SwiGLU 融合算子：前半 value，后半 gate，一次算子内完成激活与相乘。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, dim=-1)
        return F.silu(y) * x
