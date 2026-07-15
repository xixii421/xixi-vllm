"""Single-sequence contiguous KV Cache for v1."""

from __future__ import annotations

import torch
from torch import nn


class KVCache:
    """Per-layer K/V tensors shaped [max_len, num_kv_heads, head_dim]."""

    def __init__(
        self,
        num_layers: int,
        max_len: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_layers = num_layers
        self.max_len = max_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.k = torch.zeros(
            num_layers, max_len, num_kv_heads, head_dim, device=device, dtype=dtype
        )
        self.v = torch.zeros(
            num_layers, max_len, num_kv_heads, head_dim, device=device, dtype=dtype
        )

    @classmethod
    def allocate(
        cls,
        model: nn.Module,
        max_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> KVCache:
        layer0 = model.model.layers[0].self_attn
        return cls(
            num_layers=len(model.model.layers),
            max_len=max_len,
            num_kv_heads=layer0.num_kv_heads,
            head_dim=layer0.head_dim,
            device=device,
            dtype=dtype,
        )

    def write(
        self,
        layer_id: int,
        start: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Write K/V for new tokens. k/v shape [batch, seq, kv_heads, head_dim]."""
        n = k.shape[1]
        end = start + n
        if end > self.max_len:
            raise ValueError(f"KV cache overflow: need {end} slots, max_len={self.max_len}")
        self.k[layer_id, start:end] = k[0]
        self.v[layer_id, start:end] = v[0]

    def read(self, layer_id: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if length > self.max_len:
            raise ValueError(f"KV cache read overflow: length={length}, max_len={self.max_len}")
        return self.k[layer_id, :length], self.v[layer_id, :length]

    def reset(self) -> None:
        self.k.zero_()
        self.v.zero_()
