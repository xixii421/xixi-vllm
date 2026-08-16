"""Radix-tree prefix index."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .hash_util import compute_block_hash
from .radix_tree import RadixTree

if TYPE_CHECKING:
    from ..block_manager import BlockManager
    from ..sequence import Sequence


class RadixPrefixIndex:
    def __init__(self) -> None:
        self.tree = RadixTree()
        self._seq_node: dict[int, object] = {}

    def _block_hashes(self, seq: Sequence, n: int) -> list[int]:
        h = -1
        out: list[int] = []
        for i in range(n):
            h = compute_block_hash(seq.block_token_ids(i), h)
            out.append(h)
        return out

    def match(self, seq: Sequence, pool: BlockManager) -> list[int]:
        n_full = seq.num_full_blocks()
        if n_full <= 0:
            return []
        key = self._block_hashes(seq, n_full)
        cached_ids, _ = self.tree.match_prefix(key)
        hit: list[int] = []
        for i, bid in enumerate(cached_ids):
            if bid < 0 or i >= n_full:
                break
            toks = seq.block_token_ids(i)
            block = pool.blocks[bid]
            if block.hash != key[i] or block.token_ids != toks:
                break
            if bid not in pool.used_ids and bid not in pool.cached_ids:
                break
            hit.append(bid)
        return hit

    def publish(
        self,
        seq: Sequence,
        num_full_blocks: int,
        pool: BlockManager,
        *,
        start: int = 0,
    ) -> None:
        if num_full_blocks <= start:
            return
        h = -1
        if start > 0:
            h = pool.blocks[seq.block_table[start - 1]].hash
            if h == -1:
                start = 0
        for i in range(start, num_full_blocks):
            bid = seq.block_table[i]
            toks = seq.block_token_ids(i)
            h = compute_block_hash(toks, h)
            pool.blocks[bid].update(h, toks)

        key = [pool.blocks[seq.block_table[i]].hash for i in range(num_full_blocks)]
        value = list(seq.block_table[:num_full_blocks])
        matched_len, last = self.tree.insert(key, value)

        # Dedup: remap onto canonical tree blocks when a longer shared prefix exists.
        if matched_len > 0:
            tree_ids, _ = self.tree.match_prefix(key)
            for i in range(min(matched_len, len(tree_ids), num_full_blocks)):
                old_bid = seq.block_table[i]
                new_bid = tree_ids[i]
                if old_bid == new_bid:
                    continue
                pool.release_hit(old_bid)
                pool.acquire_hit(new_bid)
                seq.block_table[i] = new_bid

        old = self._seq_node.pop(seq.seq_id, None)
        if old is not None:
            self.tree.dec_ref(old)  # type: ignore[arg-type]
        self.tree.inc_ref(last)
        self._seq_node[seq.seq_id] = last

    def invalidate(self, block_id: int, pool: BlockManager) -> None:
        """Detach zero-ref tree node owning this page (real invalidate, unlike v2)."""
        del pool  # residency cleared by BlockManager before/after
        self.tree.detach_leaf_for_block(block_id)

    def release_seq(self, seq_id: int) -> None:
        node = self._seq_node.pop(seq_id, None)
        if node is not None:
            self.tree.dec_ref(node)  # type: ignore[arg-type]

    def pin_match(self, seq: Sequence, num_cached: int) -> None:
        """Pin tree node after allocate hit (call once per live seq)."""
        if num_cached <= 0:
            return
        key = self._block_hashes(seq, num_cached)
        _ids, node = self.tree.match_prefix(key)
        old = self._seq_node.pop(seq.seq_id, None)
        if old is not None:
            self.tree.dec_ref(old)  # type: ignore[arg-type]
        self.tree.inc_ref(node)
        self._seq_node[seq.seq_id] = node

    def reclaim_pages(self, num_pages: int, pool: BlockManager) -> list[int]:
        """Tree leaf-LRU reclaim: drop cold zero-ref leaves, return cached page ids.

        May return more than ``num_pages`` when a leaf holds multiple pages — the
        whole leaf must move to free together (index already detached).
        """
        if num_pages <= 0:
            return []
        out: list[int] = []
        guard = 0
        while len(out) < num_pages and guard < 64:
            guard += 1
            freed = self.tree.evict(max(1, num_pages - len(out)))
            if not freed:
                break
            for bid in freed:
                if bid in pool.cached_ids:
                    out.append(bid)
        return out
