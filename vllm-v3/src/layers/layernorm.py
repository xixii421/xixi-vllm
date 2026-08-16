from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x if residual is None else x + residual
        return F.rms_norm(residual, self.weight.shape, self.weight, self.eps), residual
