"""Unit tests for BlockManager + hash prefix + slots (CPU)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory import BlockManager, Sequence, slot_of, slots_for_range
from src.sampling_params import SamplingParams


def test_blocks_for_tokens() -> None:
    bm = BlockManager(num_blocks=32, block_size=16)
    assert bm.blocks_for_tokens(0) == 0
    assert bm.blocks_for_tokens(1) == 1
    assert bm.blocks_for_tokens(16) == 1
    assert bm.blocks_for_tokens(17) == 2
    print("  [PASS] blocks_for_tokens")


def test_allocate_and_deallocate() -> None:
    Sequence.block_size = 4
    bm = BlockManager(num_blocks=8, block_size=4)
    seq = Sequence([1, 2, 3, 4, 5])  # 2 pages
    n = bm.can_allocate(seq)
    assert n != -1
    bm.allocate(seq, n)
    assert len(seq.block_table) == 2
    assert bm.num_free == 6
    bm.deallocate(seq)
    assert bm.num_free == 8
    assert seq.block_table == []
    print("  [PASS] allocate / deallocate")


def test_may_append() -> None:
    Sequence.block_size = 4
    bm = BlockManager(num_blocks=16, block_size=4)
    seq = Sequence([10, 11, 12, 13])  # 1 full page
    bm.allocate(seq, bm.can_allocate(seq))
    assert len(seq.block_table) == 1
    # After appending one more token, len=5, 5%4==1 → need new page before decode write
    seq.append_token(14)
    assert bm.can_append(seq)
    bm.may_append(seq)
    assert len(seq.block_table) == 2
    bm.deallocate(seq)
    print("  [PASS] may_append")


def test_hash_prefix_reuse() -> None:
    Sequence.block_size = 4
    bm = BlockManager(num_blocks=16, block_size=4)
    sp = SamplingParams(max_tokens=1)

    # Seq A occupies and hashes one full block of tokens [1,2,3,4]
    a = Sequence([1, 2, 3, 4, 5, 6], sp)
    bm.allocate(a, bm.can_allocate(a))
    a.num_scheduled_tokens = 4
    a.num_cached_tokens = 0
    bm.hash_blocks(a)
    a.num_cached_tokens = 4
    shared = a.block_table[0]
    free_before = bm.num_free

    # Seq B shares the same first full block
    b = Sequence([1, 2, 3, 4, 9, 9], sp)
    n = bm.can_allocate(b)
    assert n >= 1
    bm.allocate(b, n)
    assert b.block_table[0] == shared
    assert bm.blocks[shared].ref_count == 2
    assert bm.num_free == free_before - (len(b.block_table) - 1)

    bm.deallocate(a)
    assert bm.blocks[shared].ref_count == 1
    assert shared in bm.used_block_ids

    bm.deallocate(b)
    assert bm.num_free == 16
    print("  [PASS] hash prefix reuse + ref_count")


def test_slot_mapping() -> None:
    table = [7, 2]
    assert slot_of(table, 0, 4) == 28
    assert slot_of(table, 4, 4) == 8
    assert slots_for_range(table, 2, 6, 4) == [30, 31, 8, 9]
    print("  [PASS] slot_of / slots_for_range")


def main() -> None:
    print("── BlockManager (hash prefix) ──")
    test_blocks_for_tokens()
    test_allocate_and_deallocate()
    test_may_append()
    test_hash_prefix_reuse()
    test_slot_mapping()
    print("\n✅ BlockManager tests passed.")


if __name__ == "__main__":
    main()
