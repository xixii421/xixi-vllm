"""多头 / 分组查询注意力，走 PyTorch scaled_dot_product_attention。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _sdpa_supports_gqa() -> bool:
    """探测当前 PyTorch 的 SDPA 是否接受 enable_gqa=True。"""
    try:
        q = torch.zeros(1, 2, 1, 4)
        k = torch.zeros(1, 1, 1, 4)
        F.scaled_dot_product_attention(q, k, k, is_causal=True, enable_gqa=True)
        return True
    except TypeError:
        return False


_SDPA_HAS_GQA = _sdpa_supports_gqa()


class Attention(nn.Module):
    """输入 / 输出布局均为 [B, S, H, D]；内部转成 SDPA 需要的 [B, H, S, D]。

    num_kv_heads < num_heads 时为 GQA：多个 Q 头共享同一组 K/V。
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.scaling = scaling

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # [B, S, H, D] → [B, H, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if _SDPA_HAS_GQA and self.num_kv_heads != self.num_heads:
            # 新版 SDPA 原生 GQA：不必事先 repeat K/V
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.scaling, enable_gqa=True
            )
        else:
            if self.num_kv_heads != self.num_heads:
                # 兼容路径：把每个 KV 头按组复制到与 Q 头数对齐
                n_rep = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.scaling
            )
        return out.transpose(1, 2).contiguous()
