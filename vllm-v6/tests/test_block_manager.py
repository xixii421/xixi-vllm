"""BlockManager: alloc / free / prefix cache / append / hash collision safety."""

from __future__ import annotations

import pytest

from src.scheduling.block_manager import BlockManager
from src.scheduling.sequence import Sequence
from src.sampling_params import SamplingParams


def _seq(tokens: list[int], bs: int = 8) -> Sequence:
    Sequence.block_size = bs
    s = Sequence(tokens, SamplingParams(max_tokens=4))
    return s


class TestBlockManager:
    def test_allocate_and_free(self):
        bm = BlockManager(8, 8)
        s = _seq([1] * 20)
        n = bm.can_allocate(s)
        assert n >= 0
        bm.allocate(s, n)
        assert len(s.block_table) == 3
        used = bm.num_used
        bm.deallocate(s)
        assert s.block_table == [] and bm.num_free == 8
        assert used > 0

    def test_oom(self):
        bm = BlockManager(2, 8)
        s = _seq([1] * 40)
        assert bm.can_allocate(s) == -1

    def test_prefix_cache_reuse(self):
        bm = BlockManager(16, 8)
        s1 = _seq([1] * 20)
        bm.allocate(s1, bm.can_allocate(s1))
        s1.num_scheduled_tokens = 16  # complete 2 full pages
        s1.num_cached_tokens = 0
        bm.hash_blocks(s1)
        s1.num_cached_tokens = 16

        s2 = _seq([1] * 16 + [2] * 4)
        n = bm.can_allocate(s2)
        assert n >= 2
        bm.allocate(s2, n)
        assert s2.block_table[:2] == s1.block_table[:2]
        assert s2.num_cached_tokens == 16

    def test_may_append_at_boundary(self):
        bm = BlockManager(8, 8)
        s = _seq([1] * 8)
        bm.allocate(s, 0)
        assert len(s.block_table) == 1
        # After one sample append in postprocess, len becomes 9 → %8==1
        s.append_token(99)
        assert len(s) % 8 == 1
        assert bm.can_append(s)
        bm.may_append(s)
        assert len(s.block_table) == 2

    def test_hash_collision_does_not_clobber(self):
        bm = BlockManager(8, 4)
        s1 = _seq([1, 2, 3, 4], bs=4)
        bm.allocate(s1, 0)
        s1.num_scheduled_tokens = 4
        s1.num_cached_tokens = 0
        bm.hash_blocks(s1)
        h = bm.blocks[s1.block_table[0]].hash
        owner = bm.hash_to_block_id[h]

        # Simulate another block claiming same hash with different tokens.
        s2 = _seq([9, 9, 9, 9], bs=4)
        bm.allocate(s2, 0)
        other = s2.block_table[0]
        bm.blocks[other].update(h, [9, 9, 9, 9])
        # hash_blocks path: would try to register — using public hash_blocks
        s2.num_scheduled_tokens = 4
        s2.num_cached_tokens = 0
        # Force compute_hash result to collide by patching
        orig = BlockManager.compute_hash

        def _collide(token_ids, prefix=-1):
            if token_ids == [9, 9, 9, 9]:
                return h
            return orig(token_ids, prefix)

        BlockManager.compute_hash = staticmethod(_collide)  # type: ignore[method-assign]
        try:
            bm.hash_blocks(s2)
        finally:
            BlockManager.compute_hash = staticmethod(orig)  # type: ignore[method-assign]

        assert bm.hash_to_block_id[h] == owner
        assert bm.blocks[owner].token_ids == [1, 2, 3, 4]
