""" 哈希前缀索引：dict[块哈希 → 物理块号] """

from __future__ import annotations

from typing import TYPE_CHECKING

from .hash_util import compute_block_hash

if TYPE_CHECKING:
    from ..pool import BlockPool
    from ..sequence import Sequence


class HashPrefixIndex:
    """ 从左链式扫满块，直到哈希 miss。 """

    def __init__(self) -> None:
        self._hash_to_block_id: dict[int, int] = {}

    def match(self, seq: Sequence, pool: BlockPool) -> list[int]:
        """ 返回可复用的物理块号列表（前缀，可能为空）。 """
        h = -1
        hit: list[int] = []
        for i in range(seq.num_full_blocks()):
            toks = seq.block_token_ids(i)
            h = compute_block_hash(toks, h)
            bid = self._hash_to_block_id.get(h, -1)
            if bid < 0:
                break
            block = pool.blocks[bid]
            if block.token_ids != toks:
                break
            if bid not in pool.used_ids and bid not in pool.cached_ids:
                break
            hit.append(bid)
        return hit

    def publish(self, seq: Sequence, num_full_blocks: int, pool: BlockPool) -> None:
        """ 把前 num_full_blocks 页登记进哈希表。 """
        if num_full_blocks <= 0:
            return
        h = -1
        for i in range(num_full_blocks):
            bid = seq.block_table[i]
            toks = seq.block_token_ids(i)
            h = compute_block_hash(toks, h)
            pool.blocks[bid].update(h, toks)
            old = self._hash_to_block_id.get(h, -1)
            if old < 0 or old == bid or pool.blocks[old].token_ids == toks:
                self._hash_to_block_id[h] = bid

    def invalidate(self, block_id: int, pool: BlockPool) -> None:
        """ 页被淘汰时删除哈希映射。 """
        block = pool.blocks[block_id]
        if block.hash != -1 and self._hash_to_block_id.get(block.hash) == block_id:
            del self._hash_to_block_id[block.hash]
