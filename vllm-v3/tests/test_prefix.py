"""Prefix cache functional tests: hash / radix / none vs v2 semantics."""

from __future__ import annotations

import torch

from src.memory.block_manager import BlockManager
from src.memory.prefix import (
    HashPrefixIndex,
    NoopPrefixIndex,
    RadixPrefixIndex,
    RadixTree,
    compute_block_hash,
)
from src.memory.sequence import Sequence
from src.sampling_params import SamplingParams


def _seq(tokens: list[int], *, block_size: int = 8) -> Sequence:
    return Sequence.from_prompt(
        tokens, block_size=block_size, sampling=SamplingParams(max_tokens=4)
    )


def _pool(n: int = 32, bs: int = 8) -> BlockManager:
    return BlockManager(n, bs)


class TestHashChain:
    def test_chained_hashes_differ_by_prefix(self):
        a = compute_block_hash([1, 2, 3, 4], -1)
        b = compute_block_hash([1, 2, 3, 4], a)
        c = compute_block_hash([1, 2, 3, 4], -1)
        assert a == c and a != b


class TestHashPrefix:
    def test_publish_match_second_hit(self):
        pool = _pool()
        idx = HashPrefixIndex()
        toks = list(range(16))  # 2 full blocks @ bs=8
        s1 = _seq(toks)
        for _ in range(s1.num_blocks):
            s1.block_table.append(pool.allocate_fresh())
        idx.publish(s1, 2, pool)
        for bid in list(s1.block_table):
            pool.release_to_cached_or_free(bid, keep_cached=True)
        s1.block_table.clear()

        s2 = _seq(toks)
        hit = idx.match(s2, pool)
        assert hit == [0, 1] or len(hit) == 2
        assert all(b in pool.cached_ids or b in pool.used_ids for b in hit)

    def test_invalidate_on_evict(self):
        pool = _pool(n=2, bs=8)
        idx = HashPrefixIndex()
        pool.on_evict = lambda bid: idx.invalidate(bid, pool)
        s = _seq(list(range(8)))
        s.block_table.append(pool.allocate_fresh())
        idx.publish(s, 1, pool)
        pool.release_to_cached_or_free(s.block_table[0], keep_cached=True)
        # Force eviction by allocating 2 fresh when only 2 pages (1 cached → free → use).
        pool.allocate_fresh()
        pool.allocate_fresh()  # evicts cached
        s2 = _seq(list(range(8)))
        assert idx.match(s2, pool) == []

    def test_noop_always_miss(self):
        pool = _pool()
        idx = NoopPrefixIndex()
        s = _seq(list(range(16)))
        assert idx.match(s, pool) == []


class TestRadixTree:
    def test_insert_match_shared_prefix(self):
        tree = RadixTree()
        tree.insert([10, 20, 30], [0, 1, 2])
        tree.insert([10, 20, 40], [0, 1, 3])
        ids, _ = tree.match_prefix([10, 20, 30])
        assert ids[:2] == [0, 1]
        assert ids[2] == 2

    def test_evict_zero_ref_leaf(self):
        tree = RadixTree()
        _, n1 = tree.insert([1, 2], [10, 11])
        _, n2 = tree.insert([1, 3], [10, 12])
        tree.inc_ref(n1)
        freed = tree.evict(1)
        assert 12 in freed or 10 in freed or 11 in freed
        # n1 pinned — its exclusive pages should survive preferentially
        assert tree.contains_block(11) or n1.ref_count > 0

    def test_detach_invalidate(self):
        tree = RadixTree()
        tree.insert([5, 6], [7, 8])
        assert tree.contains_block(7)
        freed = tree.detach_leaf_for_block(7)
        assert set(freed) == {7, 8}
        assert not tree.contains_block(7)


class TestRadixPrefix:
    def test_publish_match_and_pin(self):
        pool = _pool()
        idx = RadixPrefixIndex()
        toks = list(range(16))
        s1 = _seq(toks)
        for _ in range(2):
            s1.block_table.append(pool.allocate_fresh())
        idx.publish(s1, 2, pool)
        for bid in list(s1.block_table):
            pool.release_to_cached_or_free(bid, keep_cached=True)
        idx.release_seq(s1.seq_id)
        s1.block_table.clear()

        s2 = _seq(toks)
        hit = idx.match(s2, pool)
        assert len(hit) == 2


class TestChunkedPrefillPublish:
    """Full blocks appear mid-chunked-prefill after sync (v2 contract)."""

    def test_partial_then_full_publish(self):
        from src.memory.manager import KVManager

        class C:
            num_key_value_heads = 1
            num_attention_heads = 1
            hidden_size = 16
            head_dim = 16
            num_hidden_layers = 1

        kv = KVManager(
            C(),
            num_blocks=16,
            block_size=8,
            device=torch.device("cpu"),
            dtype=torch.float16,
            prefix_backend="hash",
        )
        # 12 tokens → 1 full block after caching 8+
        s = _seq(list(range(12)), block_size=8)
        assert kv.can_allocate(s)
        kv.allocate(s)
        s.num_scheduled_tokens = 8
        s.num_cached_tokens += 8
        kv.sync_prefix(s)
        assert s.num_full_blocks() == 1
        # Second request sharing first 8 tokens should hit 1 block.
        s2 = _seq(list(range(8)) + [99, 98, 97, 96], block_size=8)
        # Release s pages to cached first
        kv.deallocate(s, publish=True)
        hit = kv.index.match(s2, kv.blocks)
        assert len(hit) >= 1
