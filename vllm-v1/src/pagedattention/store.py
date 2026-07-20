""" 按页读写物理 KV """
from __future__ import annotations
from typing import Protocol
import torch


class _BlockTensorStore(Protocol):
    """ KV 物理缓冲须暴露的字段 """

    block_size: int     # 物理页面大小
    num_kv_heads: int   # 每个token的KV头数
    head_dim: int       # 每个KV头的维度
    k: torch.Tensor     # 物理页面中的K
    v: torch.Tensor     # 物理页面中的V


def _coerce_kv_token_major(
    k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """ KV 张量规范化，如果有batch维维度则只取第一个，确保其形状为 [S, Hkv, D]
        S: 序列长度
        Hkv: 每个token的KV头数
        D: 每个KV头的维度
    """
    if k.dim() == 4: # [B, S, Hkv, D]
        if k.shape[0] != 1:
            raise ValueError("PagedStore.write only supports batch size 1")
        return k[0], v[0]
    if k.dim() == 3:
        return k, v
    raise ValueError(f"unexpected k rank {k.dim()}")


def _flat_layer_views(
    pool: _BlockTensorStore, layer_id: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """ 取出指定层的一页K/V，并展成按 slot 行索引的二维视图 """
    flat_dim = pool.num_kv_heads * pool.head_dim

    # [num_blocks, B, Hkv, D] → [num_blocks * B, Hkv * D]，行号即 slot
    flat_k = pool.k[layer_id].reshape(-1, flat_dim)
    flat_v = pool.v[layer_id].reshape(-1, flat_dim)
    return flat_k, flat_v, flat_dim



class PagedStore:
    """分页 KV 存储: write+ gather """
    def __init__(self, pool: _BlockTensorStore) -> None:
        """ 绑定一块物理 pool """
        self.pool = pool

    @property
    def block_size(self) -> int:
        return self.pool.block_size

    @property
    def device(self) -> torch.device:
        return self.pool.k.device

    def write(
        self, layer_id: int, 
        slot_mapping: torch.Tensor,
        k: torch.Tensor, v: torch.Tensor,
    ) -> None:
        """ 把当前步算出来的K/V 按照 slot_mapping 写入GPU的物理KV页池 """

        # 规范化KV张量
        k, v = _coerce_kv_token_major(k, v)


        # 当前步有多少个token，就需要有多少个slot，否则报错
        n = k.shape[0]
        if slot_mapping.numel() != n:
            raise ValueError("slot number mismatch between token number")

        # 行号 = slot；一行 = 一个 token 的全部 KV head
        flat_k, flat_v, flat_dim = _flat_layer_views(self.pool, layer_id)
        src_k = k.reshape(n, flat_dim).detach() # [n, flat_dim] 
        src_v = v.reshape(n, flat_dim).detach() # detach避免梯度传播

        # 缓存不参与反向传播；index_copy_：dest[slot_i] = src[i]
        with torch.no_grad():
            flat_k.index_copy_(0, slot_mapping, src_k) # 将src_k写入flat_k的slot_mapping位置
            flat_v.index_copy_(0, slot_mapping, src_v) # 将src_v写入flat_v的slot_mapping位置

    def gather(
        self,
        layer_id: int,
        block_table: list[int],
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ 按照逻辑顺序收集前缀 [0, seq_len) 表示在当前序列已经有seq_len个token，需要收集KV """
        
        # 如果seq_len <= 0，则返回空tensor
        if seq_len <= 0:
            empty = self.pool.k.new_empty(
                0, self.pool.num_kv_heads, self.pool.head_dim
            )
            return empty, empty.clone()

        # 假设有10个tokens，物理块大小为4，则需要3个block，向上取整
        bs = self.pool.block_size
        need_blocks = (seq_len + bs - 1) // bs
        if len(block_table) < need_blocks:
            raise IndexError("not enough blocks in block_table")

        device = self.pool.k.device
        table = torch.tensor(
            block_table[:need_blocks], device=device, dtype=torch.long
        )
        # 10个tokens的下标
        positions = torch.arange(seq_len, device=device, dtype=torch.long)
        # 10个tokens的物理块号
        physical = table[positions // bs]
        # 10个tokens的页内偏移
        offsets = positions % bs

        # 根据物理块号和页内偏移，取出KV
        k = self.pool.k[layer_id, physical, offsets]
        v = self.pool.v[layer_id, physical, offsets]
        return k, v
