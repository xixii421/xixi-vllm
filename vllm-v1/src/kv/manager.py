""" KV 生命周期编排：allocate / ensure / finish + 前缀索引

典型流程:

    seq = manager.new_sequence(prompt_ids)
    manager.allocate(seq)          # 匹配前缀 + 分配余页
    ... Engine 绑 ForwardContext，Attention 用 store.write/gather ...
    manager.ensure_blocks(seq)     # decode 变长时补页
    manager.finish(seq)            # 发布满块进索引，页回 cached/free
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from ..pagedattention import PagedStore
from .pool import BlockPool
from .prefix import HashPrefixIndex, NoopPrefixIndex, RadixPrefixIndex
from .sequence import Sequence

_PrefixBackend = Literal["none", "hash", "radix"]


def _make_index(backend: _PrefixBackend):
    """ 按配置创建前缀索引后端。 """
    if backend == "hash":
        return HashPrefixIndex()
    if backend == "radix":
        return RadixPrefixIndex()
    return NoopPrefixIndex()


class KVManager:
    """ 单引擎的 KV 与前缀缓存入口。 """

    def __init__(
        self,
        model: nn.Module,
        *,
        num_blocks: int,
        block_size: int,
        prefix_backend: _PrefixBackend,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.block_size = block_size
        self.prefix_backend = prefix_backend
        self.pool = BlockPool.allocate(
            model,
            num_blocks=num_blocks,
            block_size=block_size,
            device=device,
            dtype=dtype,
        )
        self.store = PagedStore(self.pool)  # Attention 只拿 store，不拿 Sequence
        self.index = _make_index(prefix_backend)
        self.pool.on_evict = lambda bid: self.index.invalidate(bid, self.pool)
        self._seq_counter = 0
        self.last_num_cached_tokens = 0

    def new_sequence(self, token_ids: list[int]) -> Sequence:
        """ 创建尚未绑页的 Sequence。 """
        self._seq_counter += 1
        return Sequence(
            token_ids=list(token_ids),
            block_size=self.block_size,
            seq_id=self._seq_counter,
        )

    def allocate(self, seq: Sequence) -> None:
        """ 前缀命中复用页，其余 allocate_fresh；写好 block_table。 """
        seq.block_table.clear()
        seq.num_cached_tokens = 0
        need = seq.num_blocks
        if need == 0:
            self.last_num_cached_tokens = 0
            return

        # 命中过长可能导致 free 不够：逐页缩短可复用前缀
        hit = self.index.match(seq, self.pool)
        max_cached = min(len(hit), need)
        while max_cached >= 0:
            new_needed = need - max_cached
            reclaimable = sum(1 for b in hit[:max_cached] if b in self.pool.cached_ids)
            if len(self.pool.free_ids) + reclaimable >= new_needed:
                break
            max_cached -= 1
        if max_cached < 0:
            raise RuntimeError("not enough KV blocks for sequence")

        for bid in hit[:max_cached]:
            if bid in self.pool.used_ids:
                self.pool.blocks[bid].ref_count += 1
            elif bid in self.pool.cached_ids:
                self.pool.reclaim_cached(bid)
            else:
                raise RuntimeError(f"prefix hit on non-resident block {bid}")
            seq.block_table.append(bid)

        for _ in range(max_cached, need):
            seq.block_table.append(self.pool.allocate_fresh())

        seq.num_cached_tokens = max_cached * self.block_size
        self.last_num_cached_tokens = seq.num_cached_tokens

        if isinstance(self.index, RadixPrefixIndex) and max_cached > 0:
            self.index.pin_match(seq, max_cached)

    def ensure_blocks(self, seq: Sequence) -> None:
        """ decode 变长时把 block_table 补到够用。 """
        while len(seq.block_table) < seq.num_blocks:
            seq.block_table.append(self.pool.allocate_fresh())

    def finish(self, seq: Sequence) -> None:
        """ 满块发布进前缀索引；页转 cached 或 free。 """
        n_full = seq.num_full_blocks()
        if n_full > 0:
            self.index.publish(seq, n_full, self.pool)

        if isinstance(self.index, RadixPrefixIndex):
            self.index.release_seq(seq.seq_id)

        published = set(seq.block_table[:n_full]) if n_full > 0 else set()
        for bid in list(seq.block_table):
            keep = bid in published and self.prefix_backend != "none"
            if not keep and self.prefix_backend == "hash":
                self.index.invalidate(bid, self.pool)
            self.pool.release_to_cached_or_free(bid, keep_cached=keep)

        seq.block_table.clear()
        seq.num_cached_tokens = 0
