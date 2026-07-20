"""MoE token→expert align（语义对齐 vLLM ``moe_align_block_size``）。

优先走自编译 CUDA（``csrc/moe``）；不可用时回退 Python/Triton 实现。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def align_capacity(num_valid: int, num_experts: int, block_size: int) -> int:
    return int(num_valid + num_experts * (block_size - 1))


@triton.jit
def _fill_expert_ids_kernel(
    starts_ptr,
    expert_ids_ptr,
    n_post_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    n_blocks_cap,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_blocks_cap
    n_post = tl.load(n_post_ptr)
    token_pos = offs * block_size
    active = mask & (token_pos < n_post)

    expert = tl.full([BLOCK], -1, dtype=tl.int32)
    for e in tl.static_range(0, num_experts):
        s0 = tl.load(starts_ptr + e)
        s1 = tl.load(starts_ptr + e + 1)
        hit = active & (token_pos >= s0) & (token_pos < s1)
        expert = tl.where(hit, e, expert)
    tl.store(expert_ids_ptr + offs, expert, mask=mask)


def _moe_align_block_size_python(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = topk_ids.device
    flat = topk_ids.reshape(-1)
    if flat.dtype != torch.int32:
        flat = flat.to(torch.int32)
    flat = flat.contiguous()
    n_valid = flat.numel()
    if n_valid == 0:
        z = torch.zeros(0, dtype=torch.int32, device=device)
        return z, z, torch.zeros(1, dtype=torch.int32, device=device)

    counts = torch.bincount(flat, minlength=num_experts).to(torch.int32)
    padded = ((counts + block_size - 1) // block_size) * block_size
    starts = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    starts[0] = 0
    torch.cumsum(padded, dim=0, out=starts[1:])
    num_post = starts[-1:].contiguous()

    cap = align_capacity(n_valid, num_experts, block_size)
    sorted_ids = torch.full((cap,), n_valid, dtype=torch.int32, device=device)
    n_blocks_cap = (cap + block_size - 1) // block_size
    expert_ids = torch.full((n_blocks_cap,), -1, dtype=torch.int32, device=device)

    order = torch.argsort(flat, stable=True)
    sorted_experts = flat[order]
    compact = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    compact[0] = 0
    torch.cumsum(counts, dim=0, out=compact[1:])
    arange = torch.arange(n_valid, device=device, dtype=torch.int32)
    dest = starts[sorted_experts] + (arange - compact[sorted_experts])
    sorted_ids[dest.long()] = order.to(torch.int32)

    _fill_expert_ids_kernel[(triton.cdiv(n_blocks_cap, 256),)](
        starts,
        expert_ids,
        num_post,
        num_experts=num_experts,
        block_size=block_size,
        n_blocks_cap=n_blocks_cap,
        BLOCK=256,
    )
    return sorted_ids, expert_ids, num_post


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(sorted_token_ids[cap], expert_ids[cap/block], num_tokens_post_pad[1])``."""
    from .cuda_ops import moe_align_block_size_cuda, moe_cuda_available

    flat = topk_ids.reshape(-1)
    n_valid = flat.numel()
    if n_valid == 0:
        z = torch.zeros(0, dtype=torch.int32, device=topk_ids.device)
        return z, z, torch.zeros(1, dtype=torch.int32, device=topk_ids.device)

    if moe_cuda_available():
        # Match official capacity heuristic (incl. small-batch trim).
        max_num_tokens_padded = n_valid + num_experts * (block_size - 1)
        if n_valid < num_experts:
            max_num_tokens_padded = min(n_valid * block_size, max_num_tokens_padded)
        sorted_ids = torch.empty(
            (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
        )
        max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
        expert_ids = torch.empty(
            (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
        )
        num_tokens_post_pad = torch.empty(
            (1,), dtype=torch.int32, device=topk_ids.device
        )
        ids = topk_ids if topk_ids.dtype in (torch.int32, torch.int64) else topk_ids.to(
            torch.int32
        )
        moe_align_block_size_cuda(
            ids.contiguous(),
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
        )
        return sorted_ids, expert_ids, num_tokens_post_pad

    return _moe_align_block_size_python(topk_ids, block_size, num_experts)
