from __future__ import annotations
import torch
from .pool import BlockPool
from .sequence import Sequence


def write_kv(
    pool: BlockPool,
    seq: Sequence,
    layer_id: int,
    start: int,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    n = k.shape[1]
    bs = pool.block_size
    for i in range(n):
        pos = start + i
        bid = seq.block_table[pos // bs]
        off = pos % bs
        pool.k[layer_id, bid, off].copy_(k[0, i])
        pool.v[layer_id, bid, off].copy_(v[0, i])


def gather_kv(
    pool: BlockPool,
    seq: Sequence,
    layer_id: int,
    length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if length <= 0:
        empty = pool.k.new_empty(0, pool.num_kv_heads, pool.head_dim)
        return empty, empty
    bs = pool.block_size
    n_blocks = (length + bs - 1) // bs
    # Copy full blocks then slice — faster than per-token.
    tmp_k = pool.k.new_empty(n_blocks * bs, pool.num_kv_heads, pool.head_dim)
    tmp_v = pool.v.new_empty(n_blocks * bs, pool.num_kv_heads, pool.head_dim)
    for bi in range(n_blocks):
        bid = seq.block_table[bi]
        lo = bi * bs
        tmp_k[lo : lo + bs].copy_(pool.k[layer_id, bid])
        tmp_v[lo : lo + bs].copy_(pool.v[layer_id, bid])
    return tmp_k[:length], tmp_v[:length]
