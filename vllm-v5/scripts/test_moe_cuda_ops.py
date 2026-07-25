#!/usr/bin/env python3
"""Correctness: CUDA align / silu / moe_sum vs Python/Triton references."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
PHYS = int(os.environ.get("BENCH_GPU", "3"))
os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYS)


def _ref_align(topk_ids: torch.Tensor, block: int, E: int):
    # Force Python path
    os.environ["VLLM_V7_MOE_CUDA"] = "0"
    from src.layers.moe.align import moe_align_block_size
    from src.layers.moe import cuda_ops

    cuda_ops.moe_cuda_available.cache_clear()
    cuda_ops._load.cache_clear()
    return moe_align_block_size(topk_ids, block, E)


def _cuda_align(topk_ids: torch.Tensor, block: int, E: int):
    os.environ["VLLM_V7_MOE_CUDA"] = "1"
    from src.layers.moe.align import moe_align_block_size
    from src.layers.moe import cuda_ops

    cuda_ops.moe_cuda_available.cache_clear()
    cuda_ops._load.cache_clear()
    assert cuda_ops.moe_cuda_available(), "CUDA MoE ops failed to load"
    return moe_align_block_size(topk_ids, block, E)


def check_align(M: int, topk: int, E: int, block: int) -> None:
    torch.manual_seed(0)
    ids = torch.randint(0, E, (M, topk), device="cuda", dtype=torch.int32)
    s0, e0, n0 = _ref_align(ids, block, E)
    s1, e1, n1 = _cuda_align(ids, block, E)
    n0i, n1i = int(n0.item()), int(n1.item())
    assert n0i == n1i, f"num_post {n0i} vs {n1i}"
    nblocks = (n0i + block - 1) // block
    if not torch.equal(e0[:nblocks], e1[:nblocks]):
        raise AssertionError(
            f"expert_ids mismatch\n py={e0[:nblocks].tolist()}\n cu={e1[:nblocks].tolist()}"
        )
    # Token order within an expert may differ; compare per-expert multisets.
    n_valid = M * topk

    def by_expert(sorted_ids, expert_ids):
        bags = {e: [] for e in range(E)}
        for b in range(nblocks):
            exp = int(expert_ids[b].item())
            if exp < 0:
                continue
            for t in sorted_ids[b * block : (b + 1) * block].tolist():
                if t < n_valid:
                    bags[exp].append(int(t))
        return {e: sorted(v) for e, v in bags.items()}

    assert by_expert(s0, e0) == by_expert(s1, e1), "per-expert token bags mismatch"
    print(f"  align OK M={M} topk={topk} E={E} block={block} post={n0i}")


def check_silu() -> None:
    from src.layers.moe.cuda_ops import (
        moe_cuda_available,
        mul_and_silu_cuda,
        silu_and_mul_cuda,
    )

    assert moe_cuda_available()
    x = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
    # [gate|up]
    gate, up = x.chunk(2, dim=-1)
    ref = F.silu(gate.float()) * up.float()
    out = torch.empty_like(gate)
    silu_and_mul_cuda(out, x.contiguous())
    err = (out.float() - ref).abs().max().item()
    assert err < 2e-2, err
    # [up|gate]
    y = torch.cat([up, gate], dim=-1)
    ref2 = up.float() * F.silu(gate.float())
    out2 = torch.empty_like(up)
    mul_and_silu_cuda(out2, y.contiguous())
    err2 = (out2.float() - ref2).abs().max().item()
    assert err2 < 2e-2, err2
    print(f"  silu OK err={err:.4g} mul_and_silu err={err2:.4g}")


def check_sum() -> None:
    from src.layers.moe.cuda_ops import moe_cuda_available, moe_sum_cuda

    assert moe_cuda_available()
    M, K, H = 13, 4, 2048
    x = torch.randn(M, K, H, device="cuda", dtype=torch.bfloat16)
    ref = x.float().sum(dim=1)
    out = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
    moe_sum_cuda(x, out)
    err = (out.float() - ref).abs().max().item()
    assert err < 5e-2, err
    print(f"  moe_sum OK err={err:.4g}")


def check_triton_forward() -> None:
    from src.layers.moe.triton_moe import clear_triton_moe_state, triton_moe_forward

    clear_triton_moe_state()
    torch.manual_seed(1)
    M, H, I, E, K = 8, 256, 128, 16, 4
    hidden = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    w13 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, E, (M, K), device="cuda")
    weights = torch.softmax(torch.randn(M, K, device="cuda"), dim=-1).to(
        torch.bfloat16
    )
    y = triton_moe_forward(
        hidden_states=hidden,
        topk_ids=ids,
        topk_weights=weights,
        w13_weight=w13,
        w2_weight=w2,
    )
    assert y.shape == (M, H) and torch.isfinite(y.float()).all()
    print(f"  triton_moe_forward OK shape={tuple(y.shape)}")


def main() -> None:
    torch.cuda.set_device(0)
    print(f"GPU phys={PHYS} device={torch.cuda.get_device_name(0)}")
    check_silu()
    check_sum()
    for M, block in [(1, 16), (8, 16), (32, 16), (128, 16), (512, 32)]:
        check_align(M, topk=4, E=60, block=block)
    check_triton_forward()
    print("ALL PASSED")


if __name__ == "__main__":
    main()
