from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, num_kv_heads: int) -> None:
        """GQA: Grouped Query Attention"""
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.scaling = head_dim**-0.5       # 缩放因子

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # [B, S, H, D] → [B, H, S, D] 交换head和seq维度, SDPA要求head维度在前，内存连续
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 完整的注意力 scores = QK^T / scale  -> causal mask -> softmax -> @v  -> 交换head和seq维度
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scaling, enable_gqa=self.num_kv_heads != self.num_heads)
        return out.transpose(1, 2).contiguous()
