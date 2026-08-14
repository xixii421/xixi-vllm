""" 物理 KV 页池：大张量 + 每页元数据 + free/used/cached 三态 """

from __future__ import annotations

from dataclasses import dataclass,field

import torch
from torch import nn

@dataclass
class _Block:
    """ 单页元数据 """

    ref_count: int = 0 # 引用计数，用于前缀缓存共享块
    hash: int = -1
    token_ids: list[list] = field(default_factory=list)

    def update(self,h: int, token_ids: list[list]) -> None:
        """ 写入块哈希与对应 token（publish 时调用） """
        self.hash = h
        self.token_ids = list(token_ids)

    def clear_meta(self) -> None:
        """ 清空哈希元数据，准备重回 free """
        self.hash = -1
        self.token_ids = []

class BlockPool:
    """
    GPU 上固定大小的分页 KV 缓冲
    k/v 形状: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    页状态: free（空闲）/ used（请求占用）/ cached（结束后保留供前缀命中）
    """

    def __init__(
        self,
        *, # 仅限关键字参数分隔符，之后的所有形参，调用时必须用关键词传参
        num_layers: int, # transformer 层数，每一层 transformer 独立存一套 KV
        num_blocks: int, # 物理块总数
        block_size: int, # 物理块大小，token数
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.blocks = [_Block() for _ in range(self.num_blocks)] # 所有物理块的信息列表
        self.free_ids: set[int] = set(range(num_blocks))
        self.used_ids: set[int] = set()
        self.cached_ids: set[int] = set()
        self.on_evict = None
        # 淘汰 cached 页时回调，供前缀索引 invalidate
        # 通知上层全局前缀哈希索引，删除该hash条目
        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)

    @classmethod # 第一个参数不再是self 而是cls 代表类本身，需要手动创建实例可以直接调用
    def allocate(
        cls,
        model: nn.Module,
        *,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BlockPool:
        """ 按模型层数 / KV 头维创建池 """
        # 自动解析模型内部超参
        layer0 = model.model.layers[0].self_attn
        return cls(
            num_layers=len(model.model.layers),
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=layer0.num_kv_heads,
            head_dim=layer0.head_dim,
            device=device,
            dtype=dtype,
        )

    def _evict_cached(self,bid: int) -> None:
        """ 丢掉一块 cached，必要时通知前缀索引 """
        if self.on_evict is not None:
            self.on_evict(bid)
        self.cached_ids.discard(bid)
        self.blocks[bid].ref_count = 0
        self.blocks[bid].clear_meta()
        self.free_ids.add(bid)

    def allocate_fresh(self) -> int:
        """ 取一页空闲块；free 空则淘汰一块 cached """
        if not self.free_ids: # 无空闲块
            if not self.cached_ids: # 无缓存块
                raise RuntimeError("KV block pool exhausted")
            bid = next(iter(self.cached_ids))
            # 随机从缓存块取 真实场景不会随机销毁，一般是LRU取
            self._evict_cached(bid)
        bid = self.free_ids.pop()
        block = self.blocks[bid]
        block.ref_count = 1
        block.clear_meta() # 防御性二次清理，防止脏元数据残留
        self.used_ids.add(bid)
        return bid

    def reclaim_cached(self,bid: int) -> None:
        """ 前缀命中：把 cached 页重新变为 used """
        self.cached_ids.discard(bid)
        self.free_ids.discard(bid)
        self.used_ids.add(bid)
        self.blocks[bid].ref_count = 1

    def release_to_cached_or_free(self,bid: int, *,keep_cached: bool) -> None:
        """ 请求结束减引用；可保留为 cached，否则回 free """
        block = self.blocks[bid]
        block.ref_count -= 1
        if block.ref_count > 0:
            return
        self.used_ids.discard(bid) # 块没有使用者，移除
        if keep_cached and block.hash != -1:
        # keep_cached 允许走前缀缓存逻辑
        # hash不为-1 代表这块元素据有效，加入cache
            self.cached_ids.add(bid)
        else:
            block.clear_meta()
            self.cached_ids.discard(bid)
            self.free_ids.add(bid)

