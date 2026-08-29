"""Unified KV resource layer (v2-aligned): pages + GPU pool + prefix backends."""

from __future__ import annotations

from typing import Literal

import torch
from transformers import PretrainedConfig

from .block_manager import BlockManager
from .prefix import PrefixBackend, RadixPrefixIndex, make_prefix_index
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
        prefix_backend: PrefixBackend = "hash",
    ) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")

        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        self.hf_config = hf_config
        self.prefix_backend: PrefixBackend = prefix_backend
        self.num_blocks_total = num_blocks
        self.num_blocks_schedulable = num_blocks

        self.blocks = BlockManager(self.num_blocks_schedulable, block_size)
        self.index = make_prefix_index(prefix_backend)
        # Hash: page-pool LRU + invalidate. Radix: tree leaf LRU reclaim first.
        self.blocks.on_evict = lambda bid: self.index.invalidate(bid, self.blocks)
        if isinstance(self.index, RadixPrefixIndex):
            self.blocks.reclaim_fn = lambda n: self.index.reclaim_pages(n, self.blocks)
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
            2, layers, num_blocks, block_size, nkv, hd, dtype=dtype, device=device
        )
        print(
            f"KV pool: {num_blocks} blocks × {block_size} tok "
            f"({num_blocks * block_bytes / (1024**3):.2f} GiB)"
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
        gpu_memory_utilization: float, # 用户设置的GPU显存利用率上限
        reserve_bytes: int = 0,
    ) -> int:
        """ 自动计算 KV Cache 最大可用 block 数量 """
        torch.cuda.empty_cache() # 释放troch缓存中空闲显存碎片
        torch.cuda.synchronize(device) # 同步GPU
        nkv = hf.num_key_value_heads
        hd = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        layers = hf.num_hidden_layers
        block_bytes = 2 * layers * block_size * nkv * hd * dtype.itemsize
        # KV两份张量 层数 单块token数 单KVtoken头数加维度 单元素字节数
        free, total = torch.cuda.mem_get_info(device) # 设备空闲字节，总物理显存字节
        available = int(total * gpu_memory_utilization) - (total - free)
        # 预留给 CG 的字节数由调用方按实际捕获足迹传入（见 Engine._probe_cg_footprint）。
        available -= reserve_bytes
        return max(4, available // block_bytes)

    @property
    def num_free(self) -> int:
        return self.blocks.num_free

    def _fit_prefix(self, seq: Sequence) -> int:
        """预检查：计算当前序列能否复用前缀 KV 块 """
        need = seq.num_blocks
        if need == 0:
            return 0
        hit = self.index.match(seq, self.blocks)
        max_cached = min(len(hit), need)
        reclaimable = sum(
            1 for b in hit[:max_cached] if b in self.blocks.cached_ids
        )
        other_cached = len(self.blocks.cached_ids) - reclaimable
        new_needed = need - max_cached
        if self.blocks.num_free + other_cached >= new_needed:
            return max_cached
        return -1

    def can_allocate(self, seq: Sequence) -> bool:
        return self._fit_prefix(seq) >= 0

    def allocate(self, seq: Sequence) -> None:
        seq.block_table.clear()
        seq.num_cached_tokens = 0
        need = seq.num_blocks
        if need == 0:
            self.last_num_cached_tokens = 0
            return

        max_cached = self._fit_prefix(seq)
        if max_cached < 0:
            raise RuntimeError("not enough KV blocks for sequence")
        hit = self.index.match(seq, self.blocks)

        for bid in hit[:max_cached]:
            self.blocks.acquire_hit(bid)
            seq.block_table.append(bid)

        for _ in range(max_cached, need):
            seq.block_table.append(self.blocks.allocate_fresh())

        seq.num_cached_tokens = max_cached * self.block_size
        self.last_num_cached_tokens = seq.num_cached_tokens

        if isinstance(self.index, RadixPrefixIndex) and max_cached > 0:
            self.index.pin_match(seq, max_cached)

    def can_append(self, seq: Sequence) -> bool:
        return self.blocks.can_append(seq)

    def may_append(self, seq: Sequence) -> None:
        self.blocks.may_append(seq)

    def sync_prefix(self, seq: Sequence) -> None:
        """只把本步新填满的块写入前缀索引；未满块或未跨块边界则直接返回。"""
        if self.prefix_backend == "none" or seq.num_scheduled_tokens <= 0:
            return
        prev = seq.num_cached_tokens - seq.num_scheduled_tokens
        start = prev // self.block_size
        end = min(seq.num_full_blocks(), seq.num_cached_tokens // self.block_size)
        if start >= end:
            return
        self.index.publish(seq, end, self.blocks, start=start)

    def deallocate(self, seq: Sequence, *, publish: bool = True) -> None:
        n_full = seq.num_full_blocks() if publish else 0
        if publish and n_full > 0 and self.prefix_backend != "none":
            self.index.publish(seq, n_full, self.blocks)

        self.index.release_seq(seq.seq_id)

        published = set(seq.block_table[:n_full]) if n_full > 0 else set()
        for bid in list(seq.block_table):
            keep = bid in published and self.prefix_backend != "none"
            # hash：释放前摘索引；radix：先降引用，零引用后再摘树叶。
            if not keep and self.prefix_backend == "hash":
                self.index.invalidate(bid, self.blocks)
            self.blocks.release_to_cached_or_free(bid, keep_cached=keep)
            if not keep and self.prefix_backend == "radix":
                self.index.invalidate(bid, self.blocks)

        seq.block_table.clear()
        seq.num_cached_tokens = 0
