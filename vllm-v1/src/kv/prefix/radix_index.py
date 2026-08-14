""" Radix 前缀索引：块哈希序列上的压缩前缀树 """

from __future__ import annotations

from typing import TYPE_CHECKING

from kv.prefix.hash_util import compute_block_hash
from kv.prefix.radix_tree import RadixTree

if TYPE_CHECKING:
    from kv.pool import BlockPool
    from kv.sequence import Sequence

class RadixPrefixIndex:
    """ match/publish 走树；块哈希仍用 compute_block_hash """
    # 根据哈希序列生成匹配前缀树
    def __init__(self) -> None:
        self._tree = RadixTree()
        self._seq_node: dict[int,object] = {}
        # int 代表Sequence的请求id
        # object 所有类的根父类，可以是任意类，实际上是 _RadixNode 存的是对应请求的末端节点

    def _block_hashes(self,seq: Sequence, n: int) -> list[int]:
        """ 前 n 个满块的链式哈希序列（树的 key） """
        h = -1
        out: list[int] = []
        for i in range(n):
            h = compute_block_hash(seq.block_token_ids(i),h)
            out.append(h)
        return out

    def publish(self,seq: Sequence,num_full_blocks: int,pool: BlockPool) -> None:
        """ 写块元数据并插入/合并到 radix 树 """
        if num_full_blocks <= 0:
            return
        h = -1
        for i in range(num_full_blocks):
            bid = seq.block_table[i] # 物理页号
            toks = seq.block_token_ids(i) # tokens
            h = compute_block_hash(toks,h) # 计算块哈希
            pool.blocks[bid].update(h,toks) # 物理块更新信息

        key = self._block_hashes(seq,num_full_blocks) # 该请求的哈希序列
        value = list(seq.block_table[:num_full_blocks]) # 该请求的全部物理块id
        last = self._tree.insert(key,value) # 插入前缀树，返回末端节点
        old = self._seq_node.pop(seq.seq_id,None) # 取出对应请求的旧末端节点
        if old is not None: # 如果不为空，代表有
            self._tree.dec_ref(old)
            # 路径上所有节点ref_count-1 代表不再用该旧路径
            # 代表有更长的路径，也就是更长的复用前缀
        self._tree.inc_ref(last) # 增添新路径
        self._seq_node[seq.seq_id] = last # 更新字典

    def match(self,seq: Sequence, pool: BlockPool) -> list[int]:
        """ 树上最长前缀 + 元数据校验 → 可复用物理块列表 """
        n_full = seq.num_full_blocks()
        if n_full <= 0:
            return []
        key = self._block_hashes(seq,n_full) # 匹配请求的哈希序列
        cached_ids,_ = self._tree.match_prefix(key) # 得到可复用的物理地址序列
        hit: list[int] = [] # 最终复用物理地址序列
        for i,bid in enumerate(cached_ids):
            # enumerate 同时拿到下标i和元素
            if bid < 0 or i >= n_full:
                break # 无效物理块或者过长序列
            toks = seq.block_token_ids(i) # 第i个物理块对应的期望tokens
            block = pool.blocks[bid] # 实际物理块
            if block.hash != key[i] or block.token_ids != toks:
                break # 两层内容校验：块哈希不匹配或者tokens不匹配，匹配失败
            if bid not in pool.used_ids and bid not in pool.cached_ids:
                break # 物理块状态检查
            hit.append(bid)
        return hit

    def pin_match(self,seq: Sequence,num_blocks: int) -> None:
        """ allocate 命中后：为对应前缀节点加引用，供 finish 时 release_seq """
        # 只读不写树，上述match函数没有加引用，这里给树上已存在的节点增加引用
        if num_blocks <= 0:
            return
        key = self._block_hashes(seq,num_blocks)
        _ids,node = self._tree.match_prefix(key)
        self._tree.inc_ref(node)
        self._seq_node[seq.seq_id] = node

    def invalidate(self,block_id: int,pool: BlockPool) -> None:
        """ 教学实现：淘汰时不拆树，靠块元数据校验兜底 """
        return

    def release_seq(self,seq_id: int) -> None:
        """ 请求结束，松开该 seq 持有的树节点引用 """
        node = self._seq_node.pop(seq_id,None)
        if node is not None:
            self._tree.dec_ref(node)
