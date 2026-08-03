from __future__ import annotations
import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps  # 防除零

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype     

        x32 = x.float()
        if residual is not None:
            x32 = x32 + residual.float()
            residual = x32.to(orig_dtype)
        
        # RMS 归一化：x / sqrt(mean(x²) + eps) * weight
        #   x32.pow(2).mean(dim=-1, keepdim=True)  → 按特征维求均方，keepdim 保留 [B,S,1] 以便广播
        #   rsqrt(...)                              → 1/sqrt，一次算子完成开方+取倒数，省一次临时张量
        #   * self.weight                           → 可学习缩放
        #   .to(orig_dtype)                         → 结果转回原始精度
        out = (x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight).to(orig_dtype)
        return (out, residual) if residual is not None else out
