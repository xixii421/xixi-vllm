"""KV block lifecycle + prefix index (shared by control & data planes)."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from ..pagedattention import PagedStore
from .gather import gather_kv, write_kv
from .pool import BlockPool
from .prefix import HashPrefixIndex, NoopPrefixIndex, RadixPrefixIndex
from .sequence import Sequence

PrefixBackend = Literal["none", "hash", "radix"]


def _make_index(backend: PrefixBackend):
    if backend == "hash":
        return HashPrefixIndex()
    if backend == "radix":
        return RadixPrefixIndex()
    return NoopPrefixIndex()


class KVManager:
    def __init__(
        self,
        model: nn.Module,
        *,
        num_blocks: int,
        block_size: int,
        prefix_backend: PrefixBackend,
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
        self.store = PagedStore(self.pool)
        self.index = _make_index(prefix_backend)
        self.pool.on_evict = lambda bid: self.index.invalidate(bid, self.pool)
        self.last_num_cached_tokens = 0

    def _fit_prefix(self, seq: Sequence) -> int:
        """Largest prefix block count that still leaves enough free pages."""
        need = seq.num_blocks
        if need == 0:
            return 0
        hit = self.index.match(seq, self.pool)
        max_cached = min(len(hit), need)
        while max_cached >= 0:
            new_needed = need - max_cached
            reclaimable = sum(1 for b in hit[:max_cached] if b in self.pool.cached_ids)
            if len(self.pool.free_ids) + reclaimable >= new_needed:
                return max_cached
            max_cached -= 1
        return -1

    def can_allocate(self, seq: Sequence) -> bool:
        return self._fit_prefix(seq) >= 0

    def allocate(self, seq: Sequence) -> None:
        """Match prefix, bind cached blocks, allocate the rest fresh."""
        seq.block_table.clear()
        seq.num_cached_tokens = 0
        need = seq.num_blocks
        if need == 0:
            self.last_num_cached_tokens = 0
            return

        max_cached = self._fit_prefix(seq)
        if max_cached < 0:
            raise RuntimeError("not enough KV blocks for sequence")
        hit = self.index.match(seq, self.pool)

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
            key = self.index._block_hashes(seq, max_cached)  # noqa: SLF001
            _ids, node = self.index.tree.match_prefix(key)
            self.index.tree.inc_ref(node)
            self.index._seq_node[seq.seq_id] = node  # noqa: SLF001

    def can_append(self, seq: Sequence) -> bool:
        # After a token already appended at a block boundary, need one free page.
        need_new = len(seq) % self.block_size == 1
        return (not need_new) or len(self.pool.free_ids) >= 1

    def may_append(self, seq: Sequence) -> None:
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self.pool.allocate_fresh())

    def ensure_blocks(self, seq: Sequence) -> None:
        while len(seq.block_table) < seq.num_blocks:
            seq.block_table.append(self.pool.allocate_fresh())

    def sync_prefix(self, seq: Sequence) -> None:
        """Publish currently completed full blocks (for cross-request hits)."""
        n_full = seq.num_full_blocks()
        if n_full > 0 and self.prefix_backend != "none":
            self.index.publish(seq, n_full, self.pool)

    def write(
        self,
        layer_id: int,
        seq: Sequence,
        start: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        write_kv(self.pool, seq, layer_id, start, k, v)

    def gather(
        self, layer_id: int, seq: Sequence, length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return gather_kv(self.pool, seq, layer_id, length)

    def deallocate(self, seq: Sequence, *, publish: bool) -> None:
        """Release pages. ``publish=True`` keeps full blocks in prefix cache."""
        n_full = seq.num_full_blocks() if publish else 0
        if publish and n_full > 0:
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

    def finish(self, seq: Sequence) -> None:
        self.deallocate(seq, publish=True)
