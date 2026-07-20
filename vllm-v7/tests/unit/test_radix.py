from __future__ import annotations

import pytest

from src.scheduling.radix import RadixTree, prefix_len


@pytest.mark.unit
def test_prefix_len():
    assert prefix_len([1, 2, 3], [1, 2, 9]) == 2
    assert prefix_len([], [1]) == 0


@pytest.mark.unit
def test_radix_insert_match():
    tree = RadixTree()
    tree.insert([1, 2, 3, 4], [10, 11, 12, 13])
    vals, node = tree.match_prefix([1, 2, 3, 9])
    assert vals == [10, 11, 12]
    assert node is not tree.root


@pytest.mark.unit
def test_radix_share_and_evict():
    tree = RadixTree()
    tree.insert([1, 2, 3, 4], [10, 11, 12, 13])
    tree.insert([1, 2, 3, 5], [20, 21, 22, 23])
    vals, last = tree.match_prefix([1, 2, 3, 4])
    assert vals[:3] == [10, 11, 12]
    tree.inc_ref(last)
    freed = tree.evict(10)
    # referenced branch should survive; unrelated leaf may free
    assert isinstance(freed, list)
    assert 10 not in freed and 11 not in freed and 12 not in freed and 13 not in freed
    tree.dec_ref(last)


@pytest.mark.unit
def test_radix_detach_leaf_clears_stale_mapping():
    tree = RadixTree()
    tree.insert([1, 2], [10, 11])
    assert tree.contains_block(11)
    freed = tree.detach_leaf_for_block(11)
    assert freed == [10, 11]
    assert not tree.contains_block(10)
    assert not tree.contains_block(11)
    vals, _ = tree.match_prefix([1, 2])
    assert vals == []


@pytest.mark.unit
def test_block_manager_radix_share():
    from src.sampling import SamplingParams
    from src.scheduling.block_manager import BlockManager
    from src.scheduling.sequence import Sequence

    bm = BlockManager(16, 4, enable_radix=True)
    # 8 tokens → 2 full blocks eligible for caching on second seq
    toks = list(range(8))
    s1 = Sequence(toks, SamplingParams(max_tokens=1), block_size=4)
    n = bm.can_allocate(s1)
    assert n >= 0
    bm.allocate(s1, n if n > 0 else 0)
    # simulate finishing prefill of full blocks
    s1.num_scheduled_tokens = 8
    s1.num_cached_tokens = 0
    bm.hash_blocks(s1)
    s1.num_cached_tokens = 8
    pages = list(s1.block_table)
    bm.deallocate(s1)

    s2 = Sequence(toks + [99], SamplingParams(max_tokens=1), block_size=4)
    cached = bm.can_allocate(s2)
    assert cached >= 2
    bm.allocate(s2, cached)
    assert s2.block_table[:2] == pages
    assert s2.num_cached_tokens == 8
