"""Unit tests for BlockManager + slot mapping (CPU only).

Usage:
  python test/test_block_manager.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory import BlockManager, slot_of, slots_for_range


def test_blocks_for_tokens() -> None:
    bm = BlockManager(num_blocks=32, block_size=16)
    assert bm.blocks_for_tokens(0) == 0
    assert bm.blocks_for_tokens(1) == 1
    assert bm.blocks_for_tokens(16) == 1
    assert bm.blocks_for_tokens(17) == 2
    assert bm.blocks_for_tokens(33) == 3
    print("  [PASS] blocks_for_tokens")


def test_allocate_and_free() -> None:
    bm = BlockManager(num_blocks=8, block_size=4)
    t1 = bm.allocate(5)  # needs 2 pages
    assert len(t1) == 2
    assert bm.num_free == 6
    t2 = bm.allocate(12)  # needs 3 pages
    assert len(t2) == 3
    assert bm.num_free == 3
    assert set(t1).isdisjoint(set(t2))
    bm.free(t1)
    assert bm.num_free == 5
    assert t1 == []
    bm.free(t2)
    assert bm.num_free == 8
    print("  [PASS] allocate / free")


def test_may_append_idempotent() -> None:
    bm = BlockManager(num_blocks=16, block_size=4)
    table = bm.allocate(4)  # 1 page
    free_before = bm.num_free
    bm.may_append(table, 4)  # still 1 page
    assert len(table) == 1 and bm.num_free == free_before
    bm.may_append(table, 5)  # needs 2nd page
    assert len(table) == 2
    bm.may_append(table, 5)  # idempotent
    assert len(table) == 2
    bm.may_append(table, 9)  # needs 3rd
    assert len(table) == 3
    bm.free(table)
    print("  [PASS] may_append idempotent")


def test_pool_exhausted() -> None:
    bm = BlockManager(num_blocks=2, block_size=16)
    bm.allocate(32)
    try:
        bm.allocate(1)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "exhausted" in str(e).lower()
    print("  [PASS] pool exhausted")


def test_slot_mapping() -> None:
    # table maps logical block 0→7, 1→2; B=4
    table = [7, 2]
    assert slot_of(table, 0, 4) == 7 * 4 + 0
    assert slot_of(table, 3, 4) == 7 * 4 + 3
    assert slot_of(table, 4, 4) == 2 * 4 + 0
    assert slot_of(table, 7, 4) == 2 * 4 + 3
    slots = slots_for_range(table, 2, 6, 4)
    assert slots == [7 * 4 + 2, 7 * 4 + 3, 2 * 4 + 0, 2 * 4 + 1]
    print("  [PASS] slot_of / slots_for_range")


def main() -> None:
    print("── BlockManager + slot mapping ──")
    test_blocks_for_tokens()
    test_allocate_and_free()
    test_may_append_idempotent()
    test_pool_exhausted()
    test_slot_mapping()
    print("\n✅ BlockManager tests passed.")


if __name__ == "__main__":
    main()
