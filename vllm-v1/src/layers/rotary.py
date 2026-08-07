"""RoPE：预计算 cos/sin 表，按 positions 查表后旋转 Q 与 K。"""

from __future__ import annotations

import torch
from torch import nn


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """半维交错旋转：x1' = x1*cos - x2*sin，x2' = x2*cos + x1*sin。"""
    # x: [N, n_heads, rotary_dim]；cos/sin: [N, 1, rotary_dim/2]，靠广播对齐
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class RotaryEmbedding(nn.Module):
    """标准 RoPE；Qwen3 全维旋转（rotary_dim == head_size），V 不旋转。"""

    def __init__(
        self,
        head_size: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        if head_size % 2 != 0:
            raise ValueError("head_size must be even")
        self.head_size = head_size

        # inv_freq[i] = base ^ (-2i / head_size)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_size, 2, dtype=torch.float) / head_size)
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        # 缓存 [max_pos, head_size] = concat(cos, sin)，各占一半宽度
        self.register_buffer(
            "cos_sin_cache",
            torch.cat((freqs.cos(), freqs.sin()), dim=-1),
            persistent=False,
        )

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 按绝对位置从预计算表取 cos/sin，与当前序列长度对齐
        flat = positions.reshape(-1)
        cos_sin = self.cos_sin_cache.index_select(0, flat).to(dtype=query.dtype)
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

        def _apply(x: torch.Tensor) -> torch.Tensor:
            shape = x.shape
            x = x.reshape(flat.numel(), -1, self.head_size)
            return _rotate(x, cos, sin).reshape(shape)

        return _apply(query), _apply(key)
