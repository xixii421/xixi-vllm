"""Flat ``block-hash → physical page`` prefix index (legacy / fallback)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
import xxhash

if TYPE_CHECKING:
    from ..sequence import Sequence


class _BlockView(Protocol):
    hash: int
    token_ids: list[int]
    ref_count: int

    def update(self, h: int, token_ids: list[int]) -> None: ...


def compute_block_hash(token_ids: list[int], prefix: int = -1) -> int:
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.asarray(token_ids, dtype=np.int32).tobytes())
    return h.intdigest()


class HashPrefixIndex:
    """v6-style chained block hashes with a flat map."""

    def __init__(self) -> None:
        self.hash_to_block_id: dict[int, int] = {}

    def invalidate(self, block_id: int, block: _BlockView) -> None:
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]

    def match(
        self,
        seq: Sequence,
        blocks: list[_BlockView],
        used_block_ids: set[int],
        free_block_ids: set[int],
    ) -> tuple[int, int]:
        """Return ``(num_cached_blocks, num_new_blocks_needed)`` or ``(-1, -1)`` if OOM."""
        h = -1
        num_cached = 0
        num_new = seq.num_blocks
        for i in range(seq.prefix_cacheable_full_blocks()):
            token_ids = seq.block_token_ids(i)
            h = compute_block_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or blocks[block_id].token_ids != token_ids:
                break
            num_cached += 1
            if block_id in used_block_ids:
                num_new -= 1
        if len(free_block_ids) < num_new:
            return -1, -1
        return num_cached, num_new

    def bind_cached(
        self,
        seq: Sequence,
        num_cached_blocks: int,
        blocks: list[_BlockView],
        used_block_ids: set[int],
        free_block_ids: set[int],
        allocate_fresh,
    ) -> None:
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block_token_ids(i)
            h = compute_block_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = blocks[block_id]
            if block_id in used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                free_block_ids.discard(block_id)
                used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for _ in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(allocate_fresh())
        seq.num_cached_tokens = num_cached_blocks * seq.block_size

    def publish(
        self,
        seq: Sequence,
        start_block: int,
        end_block: int,
        blocks: list[_BlockView],
    ) -> None:
        if start_block >= end_block:
            return
        h = blocks[seq.block_table[start_block - 1]].hash if start_block > 0 else -1
        for i in range(start_block, end_block):
            block_id = seq.block_table[i]
            block = blocks[block_id]
            token_ids = seq.block_token_ids(i)
            h = compute_block_hash(token_ids, h)
            block.update(h, token_ids)
            old = self.hash_to_block_id.get(h, -1)
            if (
                old == -1
                or old == block_id
                or blocks[old].token_ids == token_ids
            ):
                self.hash_to_block_id[h] = block_id
