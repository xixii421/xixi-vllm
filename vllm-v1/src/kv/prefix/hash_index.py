""" 哈希前缀索引：dict[块哈希 → 物理块号] """

from __future__ import annotations

from typing import TYPE_CHECKING

from kv.prefix.hash_util import compute_block_hash

if TYPE_CHECKING:
    from ..pool import BlockPool
    from ..sequence import Sequence

class HashPrefixIndex:
    """ 从左链式扫满块，直到哈希 miss """

    def __init__(self) -> None:
        self._hash_to_block_id: dict[int, int] = {}

    def publish(self,seq: Sequence, num_full_blocks: int, pool: BlockPool) -> None:
        """ 把前 num_full_blocks 页登记进哈希表 """
        if num_full_blocks <= 0:
            return
        h = -1
        for i in range(num_full_blocks):
            bid = seq.block_table[i] # 物理块id
            toks = seq.block_token_ids(i) # token ids
            h = compute_block_hash(toks,h) # 计算链式哈希
            pool.blocks[bid].update(h,toks) # 写入块哈希以及对应的token
            old = self._hash_to_block_id.get(h,-1) # 查表
            if old < 0 or old == bid or pool.blocks[old].token_ids == toks:
                # old < 0 字典没有 key=h 直接注册
                # old == bid 字典仍然指向当前物理块，重复写入
                # 不同物理块但是token完全相同
                self._hash_to_block_id[h] = bid

    def match(self,seq: Sequence, pool: BlockPool) -> list[int]:
        """ 返回可复用的物理块号列表（前缀，可能为空） """
        h = -1
        hit: list[int] = [] # 记录可以直接复用的物理块id
        for i in range(seq.num_full_blocks()):
            toks = seq.block_token_ids(i) # 取出当前请求第i个逻辑页内部的全部token
            h = compute_block_hash(toks,h)
            bid = self._hash_to_block_id.get(h,-1) # 查表
            if bid < 0: # 查不到返回-1
                break
            block = pool.blocks[bid] # 查到了，从块池取物理块
            if block.token_ids != toks: # 防哈希膨胀，二次确认
                break
            if bid not in pool.used_ids and bid not in pool.cached_ids:
                break
            hit.append(bid)
        return hit

    def invalidate(self,block_id: int,pool: BlockPool) -> None:
        """ 页被淘汰时删除哈希映射 """
        block = pool.blocks[block_id]
        if block.hash != -1 and self._hash_to_block_id.get(block.hash) != block.id:
            # 当前物理块哈希值不为-1 且 哈希表内仍有映射 删除旧块
            del self._hash_to_block_id[block.hash]