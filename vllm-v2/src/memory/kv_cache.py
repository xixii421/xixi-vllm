from __future__ import annotations

import torch
from torch import nn


class PagedKVCache:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        # [2, layers, blocks, block_size, kv_heads, head_dim]
        self.kv = torch.zeros(
            2,
            num_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def allocate(
        cls,
        model: nn.Module,
        *,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PagedKVCache:
        layer0 = model.model.layers[0].self_attn
        return cls(
            num_layers=len(model.model.layers),
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=layer0.num_kv_heads,
            head_dim=layer0.head_dim,
            device=device,
            dtype=dtype,
        )

    @property
    def bytes_per_token(self) -> int:
        """K+V bytes for one token across all layers."""
        return (
            2
            * self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * self.kv.element_size()
        )

    @property
    def pool_nbytes(self) -> int:
        return self.kv.nbytes

    def write(
        self,
        layer_id: int,
        slot_mapping: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Scatter K/V into the pool.

        Args:
            slot_mapping: int64 ``[n]`` flat slots.
            k, v: ``[batch, seq, kv_heads, head_dim]`` (batch must be 1) or
                  ``[seq, kv_heads, head_dim]``.
        """
        if k.dim() == 4:
            if k.shape[0] != 1:
                raise ValueError("PagedKVCache.write only supports batch size 1")
            k = k[0]
            v = v[0]
        n = k.shape[0]
        if slot_mapping.numel() != n:
            raise ValueError(
                f"slot_mapping length {slot_mapping.numel()} != seq length {n}"
            )
        flat_dim = self.num_kv_heads * self.head_dim
        flat_k = self.kv[0, layer_id].reshape(-1, flat_dim)
        flat_v = self.kv[1, layer_id].reshape(-1, flat_dim)
        # Detach + no_grad: pool is not a Parameter; advanced-index
        # inplace under GradMode otherwise raises on leaf tensors.
        with torch.no_grad():
            flat_k.index_copy_(0, slot_mapping, k.reshape(n, flat_dim).detach())
            flat_v.index_copy_(0, slot_mapping, v.reshape(n, flat_dim).detach())

    def gather(
        self,
        layer_id: int,
        block_table: list[int] | torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather logical prefix ``[0, seq_len)`` into contiguous ``[seq, H, D]``."""
        if seq_len == 0:
            empty = self.kv.new_empty(0, self.num_kv_heads, self.head_dim)
            return empty, empty.clone()

        if isinstance(block_table, list):
            table = torch.tensor(block_table, device=self.device, dtype=torch.long)
        else:
            table = block_table.to(device=self.device, dtype=torch.long)

        positions = torch.arange(seq_len, device=self.device, dtype=torch.long)
        block_idx = positions // self.block_size
        offsets = positions % self.block_size
        physical = table[block_idx]
        k = self.kv[0, layer_id, physical, offsets]
        v = self.kv[1, layer_id, physical, offsets]
        return k, v

    def zero_(self) -> None:
        self.kv.zero_()
