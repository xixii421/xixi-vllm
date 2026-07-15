"""PagedKVCache write/gather round-trip tests.

Usage:
  python test/test_paged_kv.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory import BlockManager, PagedKVCache, slots_for_range


def _make_cache(
    *,
    num_layers: int = 2,
    num_blocks: int = 8,
    block_size: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 8,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> PagedKVCache:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return PagedKVCache(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )


def test_write_gather_identity() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, D = 4, 2, 8
    cache = _make_cache(block_size=B, num_kv_heads=H, head_dim=D, device=device)
    bm = BlockManager(num_blocks=cache.num_blocks, block_size=B)

    seq_len = 7  # spans 2 pages
    table = bm.allocate(seq_len)
    # Non-contiguous physical pages: reallocate scramble by freeing middle first.
    # Use explicit scramble: take blocks [3,1] if available.
    bm.free(table)
    # Force non-sequential allocation by taking one, taking another, freeing first.
    a = bm.allocate(B)  # one page
    b = bm.allocate(B)
    bm.free(a)
    c = bm.allocate(B)
    table = [b[0], c[0]]
    assert table[0] != table[1]

    k = torch.randn(1, seq_len, H, D, device=device)
    v = torch.randn(1, seq_len, H, D, device=device)
    slots = torch.tensor(
        slots_for_range(table, 0, seq_len, B), device=device, dtype=torch.long
    )
    cache.write(0, slots, k, v)
    gk, gv = cache.gather(0, table, seq_len)
    assert torch.allclose(gk, k[0]), "K gather mismatch"
    assert torch.allclose(gv, v[0]), "V gather mismatch"
    print("  [PASS] write/gather identity (non-contiguous pages)")


def test_partial_write_decode_style() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, D = 4, 1, 4
    cache = _make_cache(block_size=B, num_kv_heads=H, head_dim=D, device=device)
    bm = BlockManager(num_blocks=cache.num_blocks, block_size=B)
    table = bm.allocate(1)

    for t in range(6):
        bm.may_append(table, t + 1)
        k = torch.full((1, 1, H, D), float(t + 1), device=device)
        v = torch.full((1, 1, H, D), float(-(t + 1)), device=device)
        slots = torch.tensor(
            slots_for_range(table, t, t + 1, B), device=device, dtype=torch.long
        )
        cache.write(0, slots, k, v)
        gk, gv = cache.gather(0, table, t + 1)
        expected_k = torch.arange(1, t + 2, device=device, dtype=cache.dtype)
        expected_k = expected_k.view(t + 1, 1, 1).expand(t + 1, H, D)
        assert torch.allclose(gk, expected_k)
        assert torch.allclose(gv, -expected_k)

    print("  [PASS] decode-style append writes")


def test_bytes_accounting() -> None:
    cache = _make_cache(num_layers=3, num_blocks=10, block_size=16, num_kv_heads=4, head_dim=64)
    # 2 (K+V) * 3 * 4 * 64 * 4 bytes = 6144 if float32
    assert cache.bytes_per_token == 2 * 3 * 4 * 64 * cache.kv.element_size()
    assert cache.pool_nbytes == cache.kv.numel() * cache.kv.element_size()
    print("  [PASS] bytes_per_token / pool_nbytes")


def main() -> None:
    print("── PagedKVCache ──")
    test_write_gather_identity()
    test_partial_write_decode_style()
    test_bytes_accounting()
    print("\n✅ PagedKVCache tests passed.")


if __name__ == "__main__":
    main()
