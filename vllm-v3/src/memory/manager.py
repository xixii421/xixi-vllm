from __future__ import annotations

import torch
from transformers import PretrainedConfig

from .block_manager import BlockManager
from .sequence import Sequence


class KVManager:
    def __init__(
        self,
        hf_config: PretrainedConfig,
        *,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
        reserve_dummy_page: bool = False,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
        if reserve_dummy_page and num_blocks < 2:
            raise RuntimeError(
                f"CUDA Graph needs >= 2 KV blocks (got {num_blocks}): "
                "one page is reserved as pad/dummy"
            )

        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        self.hf_config = hf_config
        self.reserve_dummy_page = reserve_dummy_page
        self.num_blocks_total = num_blocks
        self.num_blocks_schedulable = num_blocks - 1 if reserve_dummy_page else num_blocks
        self.dummy_page_id = num_blocks - 1 if reserve_dummy_page else -1

        self.blocks = BlockManager(self.num_blocks_schedulable, block_size)
        self.kv_cache = self._alloc_pool(hf_config, num_blocks, block_size, device, dtype)
        self.last_num_cached_tokens = 0

    @staticmethod
    def _alloc_pool(
        hf: PretrainedConfig,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        nkv = hf.num_key_value_heads
        hd = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        layers = hf.num_hidden_layers
        block_bytes = 2 * layers * block_size * nkv * hd * dtype.itemsize
        kv = torch.empty(
            2,
            layers,
            num_blocks,
            block_size,
            nkv,
            hd,
            dtype=dtype,
            device=device,
        )
        tag = " [last page=CG dummy]" if num_blocks >= 2 else ""
        print(
            f"KV pool: {num_blocks} blocks × {block_size} tok "
            f"({num_blocks * block_bytes / (1024**3):.2f} GiB){tag}"
        )
        return kv

    @classmethod
    def auto_num_blocks(
        cls,
        hf: PretrainedConfig,
        *,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
        gpu_memory_utilization: float,
        reserve_cg_headroom: bool,
    ) -> int:
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        nkv = hf.num_key_value_heads
        hd = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        layers = hf.num_hidden_layers
        block_bytes = 2 * layers * block_size * nkv * hd * dtype.itemsize
        free, total = torch.cuda.mem_get_info(device)
        available = int(total * gpu_memory_utilization) - (total - free)
        if reserve_cg_headroom:
            available -= int(2.0 * 1024**3)
        return max(4, available // block_bytes)

    @property
    def num_free(self) -> int:
        return self.blocks.num_free

    def can_allocate(self, seq: Sequence) -> bool:
        return self.blocks.can_allocate(seq) != -1

    def allocate(self, seq: Sequence) -> None:
        n = self.blocks.can_allocate(seq)
        if n == -1:
            raise RuntimeError("not enough KV blocks for sequence")
        self.blocks.allocate(seq, n)
        self.last_num_cached_tokens = seq.num_cached_tokens

    def can_append(self, seq: Sequence) -> bool:
        return self.blocks.can_append(seq)

    def may_append(self, seq: Sequence) -> None:
        self.blocks.may_append(seq)

    def sync_prefix(self, seq: Sequence) -> None:
        self.blocks.hash_blocks(seq)

    def deallocate(self, seq: Sequence, *, publish: bool = True) -> None:
        del publish  # hash table retains entries until pages reused (v3 hash path).
        self.blocks.deallocate(seq)
