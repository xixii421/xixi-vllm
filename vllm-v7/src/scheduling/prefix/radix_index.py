"""Radix-tree prefix index over chained block hashes (default path)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from ..radix import RadixNode, RadixTree
from .hash_index import compute_block_hash

if TYPE_CHECKING:
    from ..sequence import Sequence


class RadixPrefixIndex:
    def __init__(self) -> None:
        self.tree = RadixTree()
        self._seq_node: dict[int, tuple[int, RadixNode]] = {}

    def block_hashes(self, seq: Sequence, num_blocks: int) -> list[int]:
        h = -1
        out: list[int] = []
        for i in range(num_blocks):
            h = compute_block_hash(seq.block_token_ids(i), h)
            out.append(h)
        return out

    def match_valid(
        self,
        seq: Sequence,
        n_full: int,
        blocks,
        used_block_ids: set[int],
        cached_block_ids: set[int],
    ) -> tuple[list[int], RadixNode]:
        if n_full <= 0:
            return [], self.tree.root
        key = self.block_hashes(seq, n_full)
        cached_ids, last = self.tree.match_prefix(key)
        valid: list[int] = []
        for i, bid in enumerate(cached_ids):
            if bid < 0:
                break
            block = blocks[bid]
            token_ids = seq.block_token_ids(i)
            if block.hash != key[i] or block.token_ids != token_ids:
                break
            if bid not in used_block_ids and bid not in cached_block_ids:
                break
            valid.append(bid)
        return valid, last

    def can_allocate(
        self,
        seq: Sequence,
        blocks,
        used_block_ids: set[int],
        cached_block_ids: set[int],
        available_slots: int,
    ) -> int:
        # Cap at text-only full pages before the first MM token.
        n_full = seq.prefix_cacheable_full_blocks()
        if n_full == 0:
            return 0 if available_slots >= seq.num_blocks else -1
        cached_ids, _ = self.match_valid(
            seq, n_full, blocks, used_block_ids, cached_block_ids
        )
        num_cached = len(cached_ids)
        slots_needed = seq.num_blocks - num_cached
        reclaim = sum(1 for bid in cached_ids if bid in cached_block_ids)
        if available_slots - reclaim < slots_needed:
            return -1
        return num_cached

    def allocate(
        self,
        seq: Sequence,
        num_cached_blocks: int,
        blocks,
        used_block_ids: set[int],
        cached_block_ids: set[int],
        reclaim_cached: Callable[[int], None],
        allocate_fresh: Callable[[], int],
    ) -> None:
        n_full = seq.prefix_cacheable_full_blocks()
        cached_ids, last_node = self.match_valid(
            seq, n_full, blocks, used_block_ids, cached_block_ids
        )
        cached_ids = cached_ids[:num_cached_blocks]
        if cached_ids:
            key = self.block_hashes(seq, len(cached_ids))
            _ids, last_node = self.tree.match_prefix(key)
            self.tree.inc_ref(last_node)
            self._seq_node[seq.seq_id] = (len(cached_ids), last_node)
        for block_id in cached_ids:
            block = blocks[block_id]
            if block_id in used_block_ids:
                block.ref_count += 1
            elif block_id in cached_block_ids:
                reclaim_cached(block_id)
            else:
                raise RuntimeError(
                    f"radix hit on non-resident block {block_id}; tree is stale"
                )
            seq.block_table.append(block_id)
        for _ in range(len(cached_ids), seq.num_blocks):
            seq.block_table.append(allocate_fresh())
        seq.num_cached_tokens = len(cached_ids) * seq.block_size

    def release_seq(self, seq_id: int) -> None:
        entry = self._seq_node.pop(seq_id, None)
        if entry is None:
            return
        _plen, node = entry
        self.tree.dec_ref(node)

    def publish(
        self,
        seq: Sequence,
        end_blocks: int,
        blocks,
        used_block_ids: set[int],
        cached_block_ids: set[int],
        free_block_ids: set[int],
        reclaim_cached: Callable[[int], None],
        deallocate_zero_ref: Callable[[int], None],
    ) -> None:
        # Caller already wrote per-block hash metadata; insert into the tree.
        if end_blocks <= 0:
            return
        key = self.block_hashes(seq, end_blocks)
        value = list(seq.block_table[:end_blocks])
        new_prefix_len, new_last = self.tree.insert(key, value)
        old_prefix_len, old_last = self._seq_node.get(seq.seq_id, (0, self.tree.root))
        if new_prefix_len > old_prefix_len and old_prefix_len != 0:
            new_ids, _ = self.tree.match_prefix(key)
            for i in range(old_prefix_len, min(new_prefix_len, len(new_ids))):
                old_bid = seq.block_table[i]
                new_bid = new_ids[i]
                if old_bid == new_bid:
                    continue
                blocks[old_bid].ref_count -= 1
                if blocks[old_bid].ref_count == 0:
                    deallocate_zero_ref(old_bid)
                if new_bid in used_block_ids:
                    blocks[new_bid].ref_count += 1
                elif new_bid in cached_block_ids:
                    reclaim_cached(new_bid)
                else:
                    blocks[new_bid].ref_count = 1
                    free_block_ids.discard(new_bid)
                    used_block_ids.add(new_bid)
                seq.block_table[i] = new_bid
        self.tree.dec_ref(old_last)
        self.tree.inc_ref(new_last)
        self._seq_node[seq.seq_id] = (len(key), new_last)

    def evict(self, need: int) -> list[int]:
        return self.tree.evict(need)

    def detach_block(self, block_id: int) -> list[int]:
        return self.tree.detach_leaf_for_block(block_id)

    def contains_block(self, block_id: int) -> bool:
        return self.tree.contains_block(block_id)
