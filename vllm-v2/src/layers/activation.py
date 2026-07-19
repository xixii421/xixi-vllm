"""SwiGLU 用的门控激活：SiLU(gate) * up。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SiluAndMul(nn.Module):
    """输入最后一维是 gate∥up 拼接结果，对半切开后做门控乘法。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up
