from __future__ import annotations
from typing import Literal
import torch
from torch import nn

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
        self.index = _make_index(prefix_backend)
        self.pool.on_evict = lambda bid: self.index.invalidate(bid, self.pool)
        self._seq_counter = 0
        self.last_num_cached_tokens = 0

    def new_sequence(self, token_ids: list[int]) -> Sequence:
        self._seq_counter += 1
        return Sequence(
            token_ids=list(token_ids),
            block_size=self.block_size,
            seq_id=self._seq_counter,
        )

    def allocate(self, seq: Sequence) -> None:
        """Match prefix, bind cached blocks, allocate the rest fresh."""
        seq.block_table.clear()
        seq.num_cached_tokens = 0
        need = seq.num_blocks
        if need == 0:
            self.last_num_cached_tokens = 0
            return

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
                # Drop index mapping? keep — still published
                if hasattr(self.index, "invalidate"):
                    pass
                self.pool.reclaim_cached(bid)
            else:
                raise RuntimeError(f"prefix hit on non-resident block {bid}")
            seq.block_table.append(bid)

        for _ in range(max_cached, need):
            seq.block_table.append(self.pool.allocate_fresh())

        seq.num_cached_tokens = max_cached * self.block_size
        self.last_num_cached_tokens = seq.num_cached_tokens

        if isinstance(self.index, RadixPrefixIndex) and max_cached > 0:
            # Hold a tree ref for the matched prefix node.
            key = self.index._block_hashes(seq, max_cached)  # noqa: SLF001
            _ids, node = self.index.tree.match_prefix(key)
            self.index.tree.inc_ref(node)
            self.index._seq_node[seq.seq_id] = node  # noqa: SLF001

    def ensure_blocks(self, seq: Sequence) -> None:
        """Grow ``block_table`` if tokens outgrew allocated pages (decode)."""
        while len(seq.block_table) < seq.num_blocks:
            seq.block_table.append(self.pool.allocate_fresh())

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

    def finish(self, seq: Sequence) -> None:
        """Publish full blocks into the prefix index; release sequence refs."""
        n_full = seq.num_full_blocks()
        # Mark hashes before / inside publish.
        if n_full > 0:
            self.index.publish(seq, n_full, self.pool)

        if isinstance(self.index, RadixPrefixIndex):
            self.index.release_seq(seq.seq_id)

        # Release all pages: keep published ones as cached, free the rest.
        published = set(seq.block_table[:n_full]) if n_full > 0 else set()
        for bid in list(seq.block_table):
            keep = bid in published and self.prefix_backend != "none"
            if keep and self.prefix_backend == "hash":
                # Ensure hash map points here.
                pass
            if not keep and self.prefix_backend == "hash":
                self.index.invalidate(bid, self.pool)
            self.pool.release_to_cached_or_free(bid, keep_cached=keep)

        seq.block_table.clear()
        seq.num_cached_tokens = 0
