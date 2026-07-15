import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        x = x.contiguous() if not x.is_contiguous() else x
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = (x * torch.rsqrt(variance + self.eps) * self.weight).to(orig_dtype)
        return (x, residual) if residual is not None else x
