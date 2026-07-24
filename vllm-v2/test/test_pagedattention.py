"""Unit: PagedStore write/gather (incl. batched decode write)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kv.pool import BlockPool
from src.pagedattention import PagedStore, slots_tensor
from src.pagedattention.slots import _slot_of


def test_slots() -> None:
    table = [7, 3]
    bs = 4
    assert _slot_of(table, 4, bs) == 3 * 4
    print("ok: slots")


def test_roundtrip_and_batch_write() -> None:
    device = torch.device("cpu")
    pool = BlockPool(
        num_layers=1,
        num_blocks=8,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        device=device,
        dtype=torch.float32,
    )
    store = PagedStore(pool)
    table = [5, 1]
    for bid in table:
        pool.free_ids.discard(bid)
        pool.used_ids.add(bid)
        pool.blocks[bid].ref_count = 1

    k = torch.randn(1, 6, 2, 8)
    v = torch.randn(1, 6, 2, 8)
    slots = slots_tensor(table, 0, 6, 4, device=device)
    store.write(0, slots, k, v)
    gk, gv = store.gather(0, table, 6)
    assert torch.allclose(gk, k[0])
    assert torch.allclose(gv, v[0])

    table2 = [2, 4]
    for bid in table2:
        pool.free_ids.discard(bid)
        pool.used_ids.add(bid)
        pool.blocks[bid].ref_count = 1
    kb = torch.randn(2, 1, 2, 8)
    vb = torch.randn(2, 1, 2, 8)
    sm = torch.tensor([2 * 4 + 0, 4 * 4 + 0], dtype=torch.long)
    store.write(0, sm, kb, vb)
    g0, _ = store.gather(0, [2], 1)
    g1, _ = store.gather(0, [4], 1)
    assert torch.allclose(g0[0], kb[0, 0])
    assert torch.allclose(g1[0], kb[1, 0])
    print("ok: write/gather + batch write")


def main() -> None:
    test_slots()
    test_roundtrip_and_batch_write()
    print("ALL_OK")


if __name__ == "__main__":
    main()
