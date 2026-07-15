from __future__ import annotations

from collections import deque

import numpy as np
import xxhash

from .radix import RadixNode, RadixTree
from .sequence import Sequence


class Block:
    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids: list[int] = []

    def update(self, h: int, token_ids: list[int]) -> None:
        self.hash = h
        self.token_ids = list(token_ids)

    def reset(self) -> None:
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """Paged KV allocator.

    When ``enable_radix=False`` (v6 behaviour): flat ``hash -> block_id`` map.
    When ``enable_radix=True``: Radix tree over *block hashes* (ullm-inspired),
    so shared prefixes map to shared physical blocks with LRU eviction.
    """

    def __init__(
        self, num_blocks: int, block_size: int, *, enable_radix: bool = False
    ) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_radix = enable_radix
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.radix = RadixTree() if enable_radix else None
        self._seq_radix: dict[int, tuple[int, RadixNode]] = {}

    @property
    def num_free(self) -> int:
        return len(self.free_block_ids)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self.free_block_ids)

    def blocks_for_tokens(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0
        return (num_tokens + self.block_size - 1) // self.block_size

    @staticmethod
    def compute_hash(token_ids: list[int], prefix: int = -1) -> int:
        if xxhash is not None:
            h = xxhash.xxh64()
            if prefix != -1:
                h.update(prefix.to_bytes(8, "little"))
            h.update(np.asarray(token_ids, dtype=np.int32).tobytes())
            return h.intdigest()
        seed = prefix if prefix != -1 else 0
        data = np.asarray(token_ids, dtype=np.int32).tobytes()
        return hash((seed, data)) & (1 << 64) - 1

    def _block_hashes(self, seq: Sequence, num_blocks: int) -> list[int]:
        h = -1
        out: list[int] = []
        for i in range(num_blocks):
            token_ids = seq.block_token_ids(i)
            h = self.compute_hash(token_ids, h)
            out.append(h)
        return out

    def _allocate_block(self) -> int:
        if not self.free_block_ids and self.radix is not None:
            freed = self.radix.evict(1)
            for bid in freed:
                if bid in self.used_block_ids and self.blocks[bid].ref_count == 0:
                    self._deallocate_block(bid)
                    if bid in self.hash_to_block_id.values():
                        for k, v in list(self.hash_to_block_id.items()):
                            if v == bid:
                                del self.hash_to_block_id[k]
        if not self.free_block_ids:
            raise RuntimeError(
                f"KV block pool exhausted: used={self.num_used}/{self.num_blocks}"
            )
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.discard(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        if getattr(seq, "has_multimodal", False) and seq.has_multimodal:
            need = seq.num_blocks
            if len(self.free_block_ids) < need:
                return -1
            return 0
        if self.enable_radix and self.radix is not None:
            return self._can_allocate_radix(seq)
        return self._can_allocate_hash(seq)

    def _can_allocate_hash(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(max(seq.num_blocks - 1, 0)):
            token_ids = seq.block_token_ids(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def _can_allocate_radix(self, seq: Sequence) -> int:
        assert self.radix is not None
        n_full = max(seq.num_blocks - 1, 0)
        if n_full == 0:
            return 0 if len(self.free_block_ids) >= seq.num_blocks else -1
        key = self._block_hashes(seq, n_full)
        cached_ids, _last = self.radix.match_prefix(key)
        num_cached = 0
        for bid in cached_ids:
            if bid < 0 or bid >= self.num_blocks:
                break
            num_cached += 1
        num_new_blocks = seq.num_blocks
        for bid in cached_ids[:num_cached]:
            if bid in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached

    def allocate(self, seq: Sequence, num_cached_blocks: int) -> None:
        assert not seq.block_table
        if self.enable_radix and self.radix is not None and not (
            getattr(seq, "has_multimodal", False) and seq.has_multimodal
        ):
            self._allocate_radix(seq, num_cached_blocks)
            return
        self._allocate_hash(seq, num_cached_blocks)

    def _allocate_hash(self, seq: Sequence, num_cached_blocks: int) -> None:
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block_token_ids(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for _ in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def _allocate_radix(self, seq: Sequence, num_cached_blocks: int) -> None:
        assert self.radix is not None
        n_full = max(seq.num_blocks - 1, 0)
        key = self._block_hashes(seq, n_full) if n_full else []
        cached_ids, last_node = self.radix.match_prefix(key) if key else ([], self.radix.root)
        cached_ids = cached_ids[:num_cached_blocks]
        if cached_ids:
            self.radix.inc_ref(last_node)
            self._seq_radix[seq.seq_id] = (len(cached_ids), last_node)
        for block_id in cached_ids:
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                if block_id in self.free_block_ids:
                    self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for _ in range(len(cached_ids), seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = len(cached_ids) * self.block_size

    def deallocate(self, seq: Sequence) -> None:
        if self.enable_radix and self.radix is not None and seq.seq_id in self._seq_radix:
            _plen, node = self._seq_radix.pop(seq.seq_id)
            self.radix.dec_ref(node)
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        need_new = len(seq) % self.block_size == 1
        return not need_new or len(self.free_block_ids) >= 1

    def may_append(self, seq: Sequence) -> None:
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence) -> None:
        if getattr(seq, "has_multimodal", False) and seq.has_multimodal:
            return
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block_token_ids(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            old = self.hash_to_block_id.get(h, -1)
            if old == -1 or old == block.block_id or self.blocks[old].token_ids == token_ids:
                self.hash_to_block_id[h] = block.block_id
        if self.enable_radix and self.radix is not None:
            self._cache_into_radix(seq, end)

    def _cache_into_radix(self, seq: Sequence, end_blocks: int) -> None:
        assert self.radix is not None
        if end_blocks <= 0:
            return
        key = self._block_hashes(seq, end_blocks)
        value = list(seq.block_table[:end_blocks])
        new_prefix_len, new_last = self.radix.insert(key, value)
        old_prefix_len, old_last = self._seq_radix.get(seq.seq_id, (0, self.radix.root))
        if new_prefix_len > old_prefix_len and old_prefix_len != 0:
            # duplicate physical blocks — keep the tree's canonical ids
            new_ids, _ = self.radix.match_prefix(key)
            for i in range(old_prefix_len, min(new_prefix_len, len(new_ids))):
                old_bid = seq.block_table[i]
                new_bid = new_ids[i]
                if old_bid == new_bid:
                    continue
                self.blocks[old_bid].ref_count -= 1
                if self.blocks[old_bid].ref_count == 0:
                    self._deallocate_block(old_bid)
                if new_bid in self.used_block_ids:
                    self.blocks[new_bid].ref_count += 1
                else:
                    self.blocks[new_bid].ref_count = 1
                    if new_bid in self.free_block_ids:
                        self.free_block_ids.remove(new_bid)
                    self.used_block_ids.add(new_bid)
                seq.block_table[i] = new_bid
        self.radix.dec_ref(old_last)
        self.radix.inc_ref(new_last)
        self._seq_radix[seq.seq_id] = (len(key), new_last)
