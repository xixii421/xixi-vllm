"""BlockManager: alloc / free / prefix cache / append / hash collision safety."""

from __future__ import annotations

import pytest

from src.scheduling.block_manager import BlockManager
from src.scheduling.sequence import Sequence
from src.sampling import SamplingParams


def _seq(tokens: list[int], bs: int = 8) -> Sequence:
    return Sequence(tokens, SamplingParams(max_tokens=4), block_size=bs)


class TestBlockManager:
    def test_allocate_and_free(self):
        bm = BlockManager(8, 8, enable_radix=True)
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
        bm = BlockManager(2, 8, enable_radix=True)
        s = _seq([1] * 40)
        assert bm.can_allocate(s) == -1

    def test_prefix_cache_reuse(self):
        bm = BlockManager(16, 8, enable_radix=True)
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
        bm = BlockManager(8, 8, enable_radix=True)
        s = _seq([1] * 8)
        bm.allocate(s, 0)
        assert len(s.block_table) == 1
        # After one sample append in postprocess, len becomes 9 → %8==1
        s.append_token(99)
        assert len(s) % 8 == 1
        assert bm.can_append(s)
        bm.may_append(s)
        assert len(s.block_table) == 2

    def test_mm_image_at_start_skips_prefix(self):
        """Image spanning page 0 → no full page is text-only → no publish/hit."""
        import torch
        from src.scheduling.sequence import MultiModalFeature

        bm = BlockManager(16, 8, enable_radix=True)
        mf = MultiModalFeature(
            modality="image",
            data=torch.zeros(1),
            start=0,
            end=4,
            num_encoder_tokens=4,
            uid="img-1",
        )
        s1 = Sequence([1] * 20, SamplingParams(max_tokens=4), mm_features=[mf], block_size=8)
        assert s1.prefix_cacheable_full_blocks() == 0
        assert bm.can_allocate(s1) == 0
        bm.allocate(s1, 0)
        s1.num_scheduled_tokens = 16
        bm.hash_blocks(s1)
        assert all(bm.blocks[b].hash == -1 for b in s1.block_table[:2])
        bm.deallocate(s1)

        s2 = Sequence([1] * 20, SamplingParams(max_tokens=4), mm_features=[mf], block_size=8)
        assert bm.can_allocate(s2) == 0

    def test_mm_text_prefix_cache_hit(self):
        """Shared tokens before the first image page can hit KV prefix."""
        import torch
        from src.scheduling.sequence import MultiModalFeature

        bm = BlockManager(16, 8, enable_radix=True)
        # Image starts at token 10 → page 0 ([0,8)) is fully text-only.
        mf1 = MultiModalFeature(
            modality="image",
            data=torch.zeros(1),
            start=10,
            end=14,
            num_encoder_tokens=4,
            uid="img-a",
        )
        tokens = [7] * 8 + [8] * 4 + [9] * 8  # 20 tokens
        s1 = Sequence(tokens, SamplingParams(max_tokens=4), mm_features=[mf1], block_size=8)
        assert s1.prefix_cacheable_full_blocks() == 1
        assert bm.can_allocate(s1) == 0
        bm.allocate(s1, 0)
        s1.num_scheduled_tokens = 16
        bm.hash_blocks(s1)
        assert bm.blocks[s1.block_table[0]].hash != -1
        assert bm.blocks[s1.block_table[1]].hash == -1  # overlaps image
        bm.deallocate(s1)

        mf2 = MultiModalFeature(
            modality="image",
            data=torch.zeros(1),
            start=10,
            end=14,
            num_encoder_tokens=4,
            uid="img-b",
        )
        s2 = Sequence(tokens, SamplingParams(max_tokens=4), mm_features=[mf2], block_size=8)
        assert bm.can_allocate(s2) == 1
        bm.allocate(s2, 1)
        assert s2.num_cached_tokens == 8

    def test_hash_collision_does_not_clobber(self):
        bm = BlockManager(8, 4, enable_radix=False)
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

    def test_hash_no_stale_hit_after_overwrite(self):
        bm = BlockManager(4, 4, enable_radix=False)
        a = _seq([1] * 8, bs=4)
        bm.allocate(a, bm.can_allocate(a))
        a.num_scheduled_tokens = 8
        a.num_cached_tokens = 0
        bm.hash_blocks(a)
        a.num_cached_tokens = 8
        bm.deallocate(a)

        b = _seq([9] * 16, bs=4)
        bm.allocate(b, bm.can_allocate(b))
        b.num_scheduled_tokens = 16
        b.num_cached_tokens = 0
        bm.hash_blocks(b)
        b.num_cached_tokens = 16
        bm.deallocate(b)

        c = _seq([1] * 8 + [2], bs=4)
        assert bm.can_allocate(c) == 0

    def test_radix_no_stale_hit_after_overwrite(self):
        bm = BlockManager(4, 4, enable_radix=True)
        a = _seq([1] * 8, bs=4)
        bm.allocate(a, bm.can_allocate(a))
        a.num_scheduled_tokens = 8
        a.num_cached_tokens = 0
        bm.hash_blocks(a)
        a.num_cached_tokens = 8
        old = list(a.block_table)
        bm.deallocate(a)
        assert set(old) <= bm.cached_block_ids

        # Fill the pool with unrelated pages, forcing physical overwrite.
        b = _seq([9] * 16, bs=4)
        n = bm.can_allocate(b)
        assert n >= 0
        bm.allocate(b, n)
        b.num_scheduled_tokens = 16
        b.num_cached_tokens = 0
        bm.hash_blocks(b)
        b.num_cached_tokens = 16
        # Original pages must no longer be radix-resident once overwritten.
        for bid in old:
            assert not bm.radix.contains_block(bid) or bid in b.block_table
        bm.deallocate(b)

        c = _seq([1] * 8 + [2], bs=4)
        hit = bm.can_allocate(c)
        assert hit == 0
        bm.allocate(c, hit)
        assert c.num_cached_tokens == 0
        assert c.block_table[:2] != old

    def test_radix_hit_reclaims_cached_pages(self):
        bm = BlockManager(8, 4, enable_radix=True)
        a = _seq([1] * 8, bs=4)
        bm.allocate(a, 0)
        a.num_scheduled_tokens = 8
        a.num_cached_tokens = 0
        bm.hash_blocks(a)
        a.num_cached_tokens = 8
        pages = list(a.block_table)
        bm.deallocate(a)
        assert set(pages) <= bm.cached_block_ids

        b = _seq([1] * 8 + [7], bs=4)
        hit = bm.can_allocate(b)
        assert hit == 2
        bm.allocate(b, hit)
        assert b.block_table[:2] == pages
        assert b.num_cached_tokens == 8
        assert pages[0] in bm.used_block_ids
        assert pages[0] not in bm.cached_block_ids

    def test_radix_eviction_under_pressure(self):
        bm = BlockManager(4, 4, enable_radix=True)
        # Cache two full pages, then allocate a cold 4-page request that needs eviction.
        a = _seq([1] * 8, bs=4)
        bm.allocate(a, 0)
        a.num_scheduled_tokens = 8
        bm.hash_blocks(a)
        a.num_cached_tokens = 8
        bm.deallocate(a)
        assert bm.cached_block_ids

        cold = _seq([3] * 16, bs=4)
        n = bm.can_allocate(cold)
        assert n == 0
        bm.allocate(cold, n)
        assert len(cold.block_table) == 4
        assert bm._available_slots() == 0
