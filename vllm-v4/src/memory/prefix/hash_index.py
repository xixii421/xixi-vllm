from __future__ import annotations

from .hash_util import compute_block_hash
from ..block_manager import BlockManager
from ..sequence import Sequence

# Sentinel IV for the first block in a chain (must match hash_util's "no prefix").
INITIAL_HASH = -1


class HashPrefixIndex:
    """Block-level APC index: chained prefix hash → candidate physical blocks.

    Uses multi-value buckets so hash collisions and duplicate physical copies of
    the same prefix remain discoverable after one copy is invalidated.
    """

    __slots__ = ("_hash_to_blocks", "_block_to_hash")

    def __init__(self) -> None:
        self._hash_to_blocks: dict[int, set[int]] = {}
        self._block_to_hash: dict[int, int] = {}

    # Back-compat for tests / benches that inspect the map directly.
    @property
    def hash_to_block_id(self) -> dict[int, int]:
        return {h: next(iter(bids)) for h, bids in self._hash_to_blocks.items() if bids}

    def _is_live(self, pool: BlockManager, bid: int) -> bool:
        return bid in pool.used_ids or bid in pool.cached_ids

    def _unlink_block(self, block_id: int) -> None:
        h = self._block_to_hash.pop(block_id, None)
        if h is None:
            return
        bucket = self._hash_to_blocks.get(h)
        if bucket is None:
            return
        bucket.discard(block_id)
        if not bucket:
            del self._hash_to_blocks[h]

    def match(self, seq: Sequence, pool: BlockManager) -> list[int]:
        h = INITIAL_HASH
        hit: list[int] = []
        for i in range(seq.prefix_cacheable_full_blocks()):
            toks = seq.block_token_ids(i)
            h = compute_block_hash(toks, h)
            bucket = self._hash_to_blocks.get(h)
            if not bucket:
                break

            chosen = -1
            for bid in tuple(bucket):
                if not self._is_live(pool, bid):
                    bucket.discard(bid)
                    self._block_to_hash.pop(bid, None)
                    continue
                block = pool.blocks[bid]
                if block.hash != h:
                    bucket.discard(bid)
                    self._block_to_hash.pop(bid, None)
                    continue
                if block.token_ids == toks:
                    chosen = bid
                    break

            if not bucket:
                self._hash_to_blocks.pop(h, None)
            if chosen < 0:
                break
            hit.append(chosen)
        return hit

    def publish(self, seq: Sequence, num_full_blocks: int, pool: BlockManager) -> None:
        n = min(
            max(num_full_blocks, 0),
            seq.prefix_cacheable_full_blocks(),
            len(seq.block_table),
        )
        if n <= 0:
            return

        h = INITIAL_HASH
        n_blocks = len(pool.blocks)
        for i in range(n):
            bid = seq.block_table[i]
            if not (0 <= bid < n_blocks):
                break
            toks = seq.block_token_ids(i)
            h = compute_block_hash(toks, h)

            # Drop any previous index entry for this physical page before rebind.
            self._unlink_block(bid)
            pool.blocks[bid].update(h, toks)

            self._block_to_hash[bid] = h
            self._hash_to_blocks.setdefault(h, set()).add(bid)

    def invalidate(self, block_id: int, pool: BlockManager) -> None:
        del pool  # reverse map is authoritative; pool kept for API compatibility
        self._unlink_block(block_id)

    def release_seq(self, seq_id: int) -> None:
        # Index is block-addressed; no per-sequence state.
        return
