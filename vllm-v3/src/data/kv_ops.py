from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """ 将算好的 k/v 按照 cache 的起始地址根据 slot_mapping 计算填入的物理位置  """
    idx = tl.program_id(0) # 当前program编号，1个program负责1个token
    slot = tl.load(slot_mapping_ptr + idx) # 取出当前token要写入的物理编号
    if slot == -1:
        return
    offsets = tl.arange(0, BLOCK_SIZE) # 一次性生成 BLOCK_SIZE 个偏移地址。同时处理这么多元素
    mask = offsets < D
    key_offsets = idx * key_stride + offsets # key起始地址加偏移地址
    value_offsets = idx * value_stride + offsets
    key = tl.load(key_ptr + key_offsets, mask=mask, other=0.0) # 数据并行加载到GPU寄存器
    value = tl.load(value_ptr + value_offsets, mask=mask, other=0.0)
    cache_offsets = slot * D + offsets # 目标地址相对于cache的偏移值
    tl.store(k_cache_ptr + cache_offsets, key, mask=mask)
    tl.store(v_cache_ptr + cache_offsets, value, mask=mask)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """ """
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()

    n, num_heads, head_dim = key.shape
    d = num_heads * head_dim
    assert slot_mapping.numel() == n

    block = triton.next_power_of_2(d)
    # 返回大于等于d的最小2的幂
    # triton的tl.arange优先要求块大小是2的幂，硬件上效率更高
    _store_kvcache_kernel[(n,)](
        # n, grid 即网络维度，用来定义启动多少个program
        key,
        key.stride(0),
        # stride 维度，跨过这个维度的1个元素，显存要调过的元素，连续情况下大小等于d
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        d,
        block,
    )
