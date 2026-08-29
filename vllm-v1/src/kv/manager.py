""" KV 生命周期编排：allocate / ensure / finish + 前缀索引

典型流程:
    seq = manager.new_sequence(prompt_ids)
    manager.allocate(seq)          # 匹配前缀 + 分配余页
    ... Engine 绑 ForwardContext，Attention 用 store.write/gather ...
    manager.ensure_blocks(seq)     # decode 变长时补页
    manager.finish(seq)            # 发布满块进索引，页回 cached/free
"""


from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from pagedattention.store import PagedStore
from kv.pool import BlockPool
from kv.prefix.hash_index import HashPrefixIndex
from kv.prefix.noop import NoopPrefixIndex
from kv.prefix.radix_index import RadixPrefixIndex
from kv.sequence import Sequence

_PrefixBackend = Literal["none","hash","radix"]

def _make_index(backend: _PrefixBackend):
    """ 按配置创建前缀索引后端 """
    if backend == "hash":
        return HashPrefixIndex()
    elif backend == "radix":
        return RadixPrefixIndex()
    else:
        return NoopPrefixIndex()

class KVManager:
    """ 单引擎的 KV 与前缀缓存入口 """

    def __init__(
        self,
        model: nn.Module, # 拿到 head_dim、num_kv_heads 等模型参数，分配 KV block 需要
        *,
        num_blocks: int, # 内存池总block数
        block_size: int, # 每个block存token数量
        prefix_backend: _PrefixBackend, # 控制索引
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.block_size = block_size
        self.prefix_backend = prefix_backend
        self.pool = BlockPool.allocate(
            model,
            num_blocks=num_blocks,
            block_size=block_size,
            device=device,
            dtype=dtype,
        ) # 按照模型创建kv缓存池
        self.store = PagedStore(self.pool) # 读写 kv缓存池
        self.index = _make_index(prefix_backend)
        self.pool.on_evict = lambda bid: self.index.invalidate(bid,self.pool)
        # 驱逐回调 内存池不够需要回收物理块bid
        self._seq_counter = 0 # 请求数量
        self.last_num_cached_tokens = 0 # 上一轮迭代一共命中缓存的token数量，用于统计前缀缓存命中率

    def new_sequence(self,token_ids: list[int]) -> Sequence:
        """ 创建尚未绑页的 Sequence """
        self._seq_counter += 1
        return Sequence(
            token_ids=list(token_ids),
            block_size=self.block_size,
            seq_id=self._seq_counter,
        )

    def allocate(self, seq: Sequence) -> None:
        """ 前缀命中复用页，其余 allocate_fresh；写好 block_table """
        seq.block_table.clear()
        seq.num_cached_tokens = 0
        need = seq.num_blocks
        if need == 0:
            self.last_num_cached_tokens = 0
            return

        # 判断当前序列是否有足够的块可以使用
        hit = self.index.match(seq, self.pool)
        max_cached = min(len(hit),need) # 计划复用块数
        reclaimable = sum(1 for b in hit[:max_cached] if b in self.pool.cached_ids)
        # 前缀已经命中的块
        other_cached = len(self.pool.cached_ids) - reclaimable
        # 除了前缀命中的其他的cached块
        new_needed = need - max_cached  # 复用之后需要的新块数
        if len(self.pool.free_ids) + other_cached < new_needed:
            raise RuntimeError("not enough KV blocks for sequence")

        for bid in hit[:max_cached]:
            if bid in self.pool.used_ids:
                self.pool.blocks[bid].ref_count += 1 # 可复用块 引用加一
            elif bid in self.pool.cached_ids:
                self.pool.reclaim_cached(bid) # 引用，转为used块
            else:
                raise RuntimeError(f"prefix hit on non-resident block {bid}")
            seq.block_table.append(bid)

        for _ in range(max_cached,need): # 分配未复用块，有free则用，没free则从cached里拿
            seq.block_table.append(self.pool.allocate_fresh())

        seq.num_cached_tokens = max_cached * self.block_size # 更新复用token数
        self.last_num_cached_tokens = seq.num_cached_tokens

        if isinstance(self.index, RadixPrefixIndex) and max_cached > 0:
            self.index.pin_match(seq, max_cached)
            # 如果索引是基树索引且有复用前缀，需要更新基树引用

    def ensure_blocks(self,seq: Sequence) -> None:
        """ decode 变长时把 block_table 补到够用 """
        while len(seq.block_table) < seq.num_blocks:
            seq.block_table.append(self.pool.allocate_fresh())

    def finish(self,seq: Sequence) -> None:
        """ 满块发布进前缀索引；页转 cached 或 free """
        # 序列seq执行结束，收尾
        n_full = seq.num_full_blocks()
        if n_full > 0: # 注册发布到前缀索引
            self.index.publish(seq,n_full,self.pool)

        if isinstance(self.index, RadixPrefixIndex):
            # 请求结束，解除该序列所有旧引用
            self.index.release_seq(seq.seq_id)

        published = set(seq.block_table[:n_full]) if n_full > 0 else set() # 进索引的满块bid
        for bid in list(seq.block_table):
            keep = bid in published and self.prefix_backend != "none"
            # bid in published 其实代表是否是完整满块
            # 之前的block_table 包含 prefix复用块，新加块，decode新加块
            if not keep and self.prefix_backend == "hash":
                self.index.invalidate(bid, self.pool) # keep不是满块则不保留清理索引
            self.pool.release_to_cached_or_free(bid, keep_cached=keep) # 所有块更新物理池状态
        seq.block_table.clear()
        seq.num_cached_tokens = 0