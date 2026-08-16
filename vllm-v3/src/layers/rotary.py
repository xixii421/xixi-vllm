from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size, "v3 RoPE requires rotary_dim == head_size"
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        # [max_pos, head_size]：前 head_size/2 = cos、后一半 = sin（对齐 FlashInfer/vLLM）
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 模型整体 .to(bf16) 会把 buffer 一并转 bf16；FlashInfer 要求 fp32，惰性转回一次
        if self.cos_sin_cache.dtype != torch.float32:
            self.cos_sin_cache = self.cos_sin_cache.float()
        return torch.ops.vllm.rotary_embedding(
            query, key, positions, self.cos_sin_cache, self.head_size
        )
