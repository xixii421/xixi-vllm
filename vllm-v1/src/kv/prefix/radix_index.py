""" Radix 前缀索引：块哈希序列上的压缩前缀树 """

from __future__ import annotations

from typing import TYPE_CHECKING

from .hash_util import compute_block_hash
from .radix_tree import RadixTree

if TYPE_CHECKING:
    from ..pool import BlockPool
    from ..sequence import Sequence


class RadixPrefixIndex:
    """ match/publish 走树；块哈希仍用 compute_block_hash。 """

    def __init__(self) -> None:
        self._tree = RadixTree()
        self._seq_node: dict[int, object] = {}

    def _block_hashes(self, seq: Sequence, n: int) -> list[int]:
        """ 前 n 个满块的链式哈希序列（树的 key）。 """
        h = -1
        out: list[int] = []
        for i in range(n):
            h = compute_block_hash(seq.block_token_ids(i), h)
            out.append(h)
        return out

    def match(self, seq: Sequence, pool: BlockPool) -> list[int]:
        """ 树上最长前缀 + 元数据校验 → 可复用物理块列表。 """
        n_full = seq.num_full_blocks()
        if n_full <= 0:
            return []
        key = self._block_hashes(seq, n_full)
        cached_ids, _ = self._tree.match_prefix(key)
        hit: list[int] = []
        for i, bid in enumerate(cached_ids):
            if bid < 0 or i >= n_full:
                break
            toks = seq.block_token_ids(i)
            block = pool.blocks[bid]
            if block.hash != key[i] or block.token_ids != toks:
                break
            if bid not in pool.used_ids and bid not in pool.cached_ids:
                break
            hit.append(bid)
        return hit

    def publish(self, seq: Sequence, num_full_blocks: int, pool: BlockPool) -> None:
        """ 写块元数据并插入/合并到 radix 树。 """
        if num_full_blocks <= 0:
            return
        h = -1
        for i in range(num_full_blocks):
            bid = seq.block_table[i]
            toks = seq.block_token_ids(i)
            h = compute_block_hash(toks, h)
            pool.blocks[bid].update(h, toks)
        key = self._block_hashes(seq, num_full_blocks)
        value = list(seq.block_table[:num_full_blocks])
        last = self._tree.insert(key, value)
        old = self._seq_node.pop(seq.seq_id, None)
        if old is not None:
            self._tree.dec_ref(old)  # type: ignore[arg-type]
        self._tree.inc_ref(last)
        self._seq_node[seq.seq_id] = last

    def pin_match(self, seq: Sequence, num_blocks: int) -> None:
        """ allocate 命中后：为对应前缀节点加引用，供 finish 时 release_seq。 """
        if num_blocks <= 0:
            return
        key = self._block_hashes(seq, num_blocks)
        _ids, node = self._tree.match_prefix(key)
        self._tree.inc_ref(node)
        self._seq_node[seq.seq_id] = node

    def invalidate(self, block_id: int, pool: BlockPool) -> None:
        """ 教学实现：淘汰时不拆树，靠块元数据校验兜底。 """
        return

    def release_seq(self, seq_id: int) -> None:
        """ 请求结束，松开该 seq 持有的树节点引用。 """
        node = self._seq_node.pop(seq_id, None)
        if node is not None:
            self._tree.dec_ref(node)  # type: ignore[arg-type]
