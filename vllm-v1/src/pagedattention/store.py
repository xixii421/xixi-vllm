""" 按页读写物理 KV 页池 """

from __future__ import annotations

import torch

class PagedStore:
    """ 分页 KV 存储：write（本步 K/V 写入 slot）+ gather（按页表读回历史）。

    物理布局 [num_layers, num_blocks, block_size, num_kv_heads, head_dim]。
    """

    def __init__(self,pool) -> None:
        """ 绑定一块物理页池。 """
        self.pool = pool

    @property
    def block_size(self) -> int:
        return self.pool.block_size

    @property
    def device(self) -> torch.device:
        return self.pool.k.device

    def write(
        self,
        layer_id: int,
        slot_mapping: torch.Tensor, # slots层给出的对应token的物理信息
        k: torch.Tensor,
        v: torch.Tensor, # 本步算出的新的kv
    ) -> None:
        """ 把当前步算出的 K/V（[B, S, Hkv, D]，B 固定为 1）按 slot_mapping 写入页池 """
        k = k[0] # 取第0个batch得到 [S,Hkv,D]
        v = v[0]
        n = k.shape[0] # 当前要写入的token数目，S
        if slot_mapping.numel() != n: # 行数必须和S一致
            raise ValueError("slot number mismatch between token number")

        # 展成按 slot 行索引的二维视图：行号 = slot，一行 = 一个 token 的全部 KV head
        flat_dim = self.pool.num_kv_heads * self.pool.head_dim # kv头打成一维
        flat_k = self.pool.k[layer_id].reshape(-1, flat_dim)
        flat_v = self.pool.v[layer_id].reshape(-1, flat_dim)
        # 把[S,Hkv,D] 重塑为 [S,flat_dim]

        # 缓存不参与反向传播；index_copy_：dest[slot_i] = src[i]
        with torch.no_grad():
            flat_k.index_copy_(0, slot_mapping, k.reshape(n, flat_dim))
            flat_v.index_copy_(0, slot_mapping, v.reshape(n, flat_dim))
            # 按照给定的slot行数，以第0维执行拷贝操作，也就是一整行复制
            # 将新算出来的kv按照slot行数拷贝进对应的内存池中
            # 此时 k.reshape(n, flat_dim).shape = [S,flat_dim]

    def gather(
        self,
        layer_id: int,
        block_table: list[int],
        seq_len: int, # 读取KV的token数
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ 按逻辑顺序读回前缀 [0, seq_len) 的 K/V """
        # 函数调用举例：
        # 读请求1 gather(layer_id, block_table=[5, 9], seq_len=21)
        # 读请求2 gather(layer_id, block_table=[3, 7, 11], seq_len=35)
        if seq_len <= 0: # 没有token需要读
            empty = self.pool.k.new_empty(
                0,self.pool.num_kv_heads,self.pool.head_dim
            ) # 基于self.pool.k 得到空张量 此时shape=[0,num_kv_heads,head_dim]
            return empty, empty.clone() # kv都返回空张量

        bs = self.pool.block_size
        need_block = (seq_len + bs -1) // bs # 需要读取的物理块数
        if len(block_table) < need_block:
            raise IndexError("not enough blocks in block_table")

        device = self.pool.k.device
        table = torch.tensor(block_table[:need_block],device=device,dtype=torch.long)
        positions = torch.arange(seq_len,device=device,dtype=torch.long)
        physical = table[positions // bs] # 每个 token 的物理块号
        offsets = positions % bs # 每个 token 的页内偏移
        k = self.pool.k[layer_id,physical,offsets] # 返回的 k.shape=[seq_len, num_kv_heads, head_dim]
        v = self.pool.v[layer_id,physical,offsets]
        return k,v