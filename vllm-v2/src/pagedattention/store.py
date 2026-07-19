from __future__ import annotations
from typing import Protocol
import torch


class BlockTensorStore(Protocol):
    block_size: int
    num_kv_heads: int
    head_dim: int
    k: torch.Tensor  # [L, N_blocks, B, Hkv, D]
    v: torch.Tensor


class PagedStore:
    def __init__(self, pool: BlockTensorStore) -> None:
        self.pool = pool

    @property
    def block_size(self) -> int:
        return self.pool.block_size

    @property
    def device(self) -> torch.device:
        return self.pool.k.device

    def write(
        self,
        layer_id: int,
        slot_mapping: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:

        # [B, S, H, D] (decode batch) or [1, S, H, D] / [S, H, D]
        if k.dim() == 4:
            b, s, h, d = k.shape
            k = k.reshape(b * s, h, d)
            v = v.reshape(b * s, h, d)
        elif k.dim() != 3:
            raise ValueError(f"unexpected k rank {k.dim()}")

        n = k.shape[0]
        if slot_mapping.numel() != n:
            raise ValueError(
                f"slot_mapping length {slot_mapping.numel()} != seq length {n}"
            )
        flat_dim = self.pool.num_kv_heads * self.pool.head_dim
        flat_k = self.pool.k[layer_id].reshape(-1, flat_dim)
        flat_v = self.pool.v[layer_id].reshape(-1, flat_dim)
        with torch.no_grad():
            flat_k.index_copy_(0, slot_mapping, k.reshape(n, flat_dim).detach())
            flat_v.index_copy_(0, slot_mapping, v.reshape(n, flat_dim).detach())

    def gather(
        self,
        layer_id: int,
        block_table: list[int],
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather logical prefix ``[0, seq_len)`` → contiguous ``[seq_len, Hkv, D]``."""
        if seq_len <= 0:
            empty = self.pool.k.new_empty(0, self.pool.num_kv_heads, self.pool.head_dim)
            return empty, empty.clone()

        bs = self.pool.block_size
        need_blocks = (seq_len + bs - 1) // bs
        if len(block_table) < need_blocks:
            raise IndexError(
                f"seq_len={seq_len} needs {need_blocks} blocks, "
                f"block_table has {len(block_table)}"
            )

        device = self.pool.k.device
        table = torch.tensor(block_table[:need_blocks], device=device, dtype=torch.long)
        positions = torch.arange(seq_len, device=device, dtype=torch.long)
        physical = table[positions // bs]
        offsets = positions % bs
        k = self.pool.k[layer_id, physical, offsets]
        v = self.pool.v[layer_id, physical, offsets]
        return k, v
