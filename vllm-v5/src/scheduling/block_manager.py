"""Paged KV block allocator + pluggable prefix index.

Public knob is ``enable_radix`` (default True). Internally maps to radix vs
flat-hash backends under ``prefix/``.
"""

from __future__ import annotations

from ..utils.profiler import Timer
from .prefix import HashPrefixIndex, RadixPrefixIndex
from .prefix.hash_index import compute_block_hash
from .sequence import Sequence


class Block:
    __slots__ = ("block_id", "ref_count", "hash", "token_ids")

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
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        *,
        enable_radix: bool = True,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_radix = bool(enable_radix)
        # Internal label for debugging / tests.
        self.prefix_mode = "radix" if self.enable_radix else "hash"
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: set[int] = set(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.cached_block_ids: set[int] = set()
        self._hash = HashPrefixIndex()
        self._radix = RadixPrefixIndex() if self.enable_radix else None
        # Compat for tests that inspect ``hash_to_block_id`` / ``radix``
        self.hash_to_block_id = self._hash.hash_to_block_id
        self.radix = self._radix.tree if self._radix is not None else None

    @property
    def num_free(self) -> int:
        return len(self.free_block_ids) + len(self.cached_block_ids)

    @property
    def num_used(self) -> int:
        return self.num_blocks - self.num_free

    def blocks_for_tokens(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0
        return (num_tokens + self.block_size - 1) // self.block_size

    @staticmethod
    def compute_hash(token_ids: list[int], prefix: int = -1) -> int:
        return compute_block_hash(token_ids, prefix)

    def _available_slots(self) -> int:
        return len(self.free_block_ids) + len(self.cached_block_ids)

    def _move_to_free(self, block_id: int) -> None:
        self.used_block_ids.discard(block_id)
        self.cached_block_ids.discard(block_id)
        block = self.blocks[block_id]
        self._hash.invalidate(block_id, block)
        block.hash = -1
        block.token_ids = []
        block.ref_count = 0
        self.free_block_ids.add(block_id)

    def _detach_cached_segment(self, block_id: int) -> None:
        if self._radix is None:
            self._move_to_free(block_id)
            return
        freed = self._radix.detach_block(block_id)
        if not freed:
            self._move_to_free(block_id)
            return
        for bid in freed:
            self._move_to_free(bid)

    def _reclaim_cached_for_reuse(self, block_id: int) -> None:
        assert block_id in self.cached_block_ids
        self.cached_block_ids.discard(block_id)
        self.used_block_ids.add(block_id)
        self.blocks[block_id].ref_count = 1

    def _allocate_block(self) -> int:
        if not self.free_block_ids:
            self._ensure_free_block()
        if not self.free_block_ids:
            raise RuntimeError(
                f"KV block pool exhausted: used={self.num_used}/{self.num_blocks}"
            )
        block_id = self.free_block_ids.pop()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        assert block_id not in self.cached_block_ids
        self._hash.invalidate(block_id, block)
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _ensure_free_block(self) -> None:
        if self._radix is not None:
            need = max(1, 1 - len(self.free_block_ids))
            for bid in self._radix.evict(need):
                self._move_to_free(bid)
            if self.free_block_ids:
                return
        if self.cached_block_ids:
            self._detach_cached_segment(next(iter(self.cached_block_ids)))

    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.discard(block_id)
        block = self.blocks[block_id]
        if self.enable_radix and block.hash != -1:
            self.cached_block_ids.add(block_id)
            return
        self.free_block_ids.add(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        with Timer("sched.kv_can_allocate"):
            if self._radix is not None:
                return self._radix.can_allocate(
                    seq,
                    self.blocks,
                    self.used_block_ids,
                    self.cached_block_ids,
                    self._available_slots(),
                )
            n_cached, _n_new = self._hash.match(
                seq, self.blocks, self.used_block_ids, self.free_block_ids
            )
            return n_cached

    def allocate(self, seq: Sequence, num_cached_blocks: int) -> None:
        with Timer("sched.kv_allocate"):
            assert not seq.block_table
            if self._radix is not None:
                self._radix.allocate(
                    seq,
                    num_cached_blocks,
                    self.blocks,
                    self.used_block_ids,
                    self.cached_block_ids,
                    self._reclaim_cached_for_reuse,
                    self._allocate_block,
                )
                return
            self._hash.bind_cached(
                seq,
                num_cached_blocks,
                self.blocks,
                self.used_block_ids,
                self.free_block_ids,
                self._allocate_block,
            )

    def deallocate(self, seq: Sequence) -> None:
        if self._radix is not None:
            self._radix.release_seq(seq.seq_id)
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        need_new = len(seq) % self.block_size == 1
        return not need_new or self._available_slots() >= 1

    def may_append(self, seq: Sequence) -> None:
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence) -> None:
        """Publish newly filled full pages into the active prefix index.

        Multimodal sequences only publish pages entirely before the first
        image token (shared text prefix); image-overlapping pages stay private.
        Publish end is ``filled_full_pages``, not ``num_blocks - 1`` (match cap).
        """
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        first = seq.first_mm_token
        if first is not None:
            end = min(end, int(first) // self.block_size)
        if start >= end:
            return
        # Always materialize per-block hash metadata (needed by both backends).
        self._hash.publish(seq, start, end, self.blocks)
        if self._radix is not None:
            self._radix.publish(
                seq,
                end,
                self.blocks,
                self.used_block_ids,
                self.cached_block_ids,
                self.free_block_ids,
                self._reclaim_cached_for_reuse,
                self._deallocate_block,
            )
