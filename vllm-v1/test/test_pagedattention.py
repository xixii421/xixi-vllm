from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kv.pool import BlockPool
from src.pagedattention import PagedStore, slots_tensor
from src.pagedattention.slots import _slot_of, _slots_for_range


def test_slot_of_and_range() -> None:
    table = [7, 3]
    bs = 4
    assert _slot_of(table, 0, bs) == 7 * 4 + 0
    assert _slot_of(table, 3, bs) == 7 * 4 + 3
    assert _slot_of(table, 4, bs) == 3 * 4 + 0
    assert _slots_for_range(table, 2, 6, bs) == [
        7 * 4 + 2,
        7 * 4 + 3,
        3 * 4 + 0,
        3 * 4 + 1,
    ]


def test_write_gather_roundtrip() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pool = BlockPool(
        num_layers=2,
        num_blocks=8,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        device=device,
        dtype=dtype,
    )
    store = PagedStore(pool)
    table = [5, 1, 3]
    for bid in table:
        if bid in pool.free_ids:
            pool.free_ids.remove(bid)
        pool.used_ids.add(bid)
        pool.blocks[bid].ref_count = 1

    seq_len = 10  # spans 3 blocks of size 4
    k = torch.randn(1, seq_len, 2, 8, device=device, dtype=dtype)
    v = torch.randn(1, seq_len, 2, 8, device=device, dtype=dtype)
    slots = slots_tensor(table, 0, seq_len, pool.block_size, device=device)
    store.write(0, slots, k, v)
    gk, gv = store.gather(0, table, seq_len)
    assert torch.allclose(gk, k[0], atol=0, rtol=0)
    assert torch.allclose(gv, v[0], atol=0, rtol=0)

    pos = 7
    k1 = torch.randn(1, 1, 2, 8, device=device, dtype=dtype)
    v1 = torch.randn(1, 1, 2, 8, device=device, dtype=dtype)
    slots1 = slots_tensor(table, pos, pos + 1, pool.block_size, device=device)
    store.write(1, slots1, k1, v1)
    gk1, gv1 = store.gather(1, table, pos + 1)
    assert torch.allclose(gk1[pos], k1[0, 0], atol=0, rtol=0)
    assert torch.allclose(gv1[pos], v1[0, 0], atol=0, rtol=0)


def main() -> None:
    test_slot_of_and_range()
    print("ok: slots")
    test_write_gather_roundtrip()
    print("ok: write/gather roundtrip")
    print("ALL_OK")


if __name__ == "__main__":
    main()
