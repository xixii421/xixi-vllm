#!/usr/bin/env python3
"""Per-stage microbench of v7 Triton MoE (one layer).

Breaks: router | prepare/align | GEMM1 | silu(+layout) | GEMM2 | moe_sum
Also reports pad inflation (num_post / (M*topk)).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.layers.activation import silu_and_mul, silu_and_mul_up_gate  # noqa: E402
from src.layers.moe.align import moe_align_block_size  # noqa: E402
from src.layers.moe.triton_moe import (  # noqa: E402
    _config,
    _gemm,
    _moe_sum,
    _prepare,
    clear_triton_moe_state,
)

PHYS = int(os.environ.get("BENCH_GPU", "3"))
LENGTHS = [1, 8, 32, 128, 512, 1024]
WARMUP = int(os.environ.get("WARMUP", "10"))
ITERS = int(os.environ.get("ITERS", "50"))

# Qwen1.5-MoE-A2.7B
H = 2048
I = 1408
E = 60
TOPK = 4
DTYPE = torch.bfloat16


def _sync() -> None:
    torch.cuda.synchronize()


def timed_ms(fn, *, iters: int) -> float:
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1000.0 / iters


def pad_stats(topk_ids: torch.Tensor, block_m: int) -> dict:
    M = topk_ids.shape[0]
    sorted_ids, expert_ids, num_post, naive, grid_tokens = _prepare(
        topk_ids, M=M, top_k=TOPK, E=E, block_m=block_m
    )
    n_valid = M * TOPK
    post = int(num_post.item())
    return {
        "naive": naive,
        "block_m": block_m,
        "n_valid": n_valid,
        "num_post": post,
        "grid_tokens": grid_tokens,
        "pad_ratio": post / max(n_valid, 1),
        "cap": None if sorted_ids is None else int(sorted_ids.numel()),
        "n_blocks": int(expert_ids.numel()),
    }


def bench_one(M: int) -> dict:
    clear_triton_moe_state()
    device = torch.device("cuda")
    hidden = torch.randn(M, H, device=device, dtype=DTYPE)
    w13 = torch.randn(E, 2 * I, H, device=device, dtype=DTYPE)
    w2 = torch.randn(E, H, I, device=device, dtype=DTYPE)
    gate_w = torch.randn(E, H, device=device, dtype=DTYPE)

    # Warm routing to get ids; also warmup kernels.
    def route():
        logits = F.linear(hidden, gate_w)
        scores = F.softmax(logits.float(), dim=-1)
        w, ids = torch.topk(scores, TOPK, dim=-1)
        return w.to(DTYPE), ids

    for _ in range(3):
        tw, ti = route()
        from src.layers.moe.triton_moe import triton_moe_forward

        _ = triton_moe_forward(
            hidden_states=hidden,
            topk_ids=ti,
            topk_weights=tw,
            w13_weight=w13,
            w2_weight=w2,
        )
    _sync()

    topk_weights, topk_ids = route()
    block_m, block_n, block_k, warps, stages = _config(M, E, TOPK)
    stats = pad_stats(topk_ids, block_m)

    # Stage lambdas operate on fresh prepares where needed.
    def stage_router():
        logits = F.linear(hidden, gate_w)
        scores = F.softmax(logits.float(), dim=-1)
        torch.topk(scores, TOPK, dim=-1)

    def stage_prepare():
        _prepare(topk_ids, M=M, top_k=TOPK, E=E, block_m=block_m)

    sorted_ids, expert_ids, num_post, naive, grid_tokens = _prepare(
        topk_ids, M=M, top_k=TOPK, E=E, block_m=block_m
    )
    num_valid = M * TOPK

    def stage_gemm1():
        return _gemm(
            hidden,
            w13,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_post=num_post,
            num_valid=num_valid,
            row_div=TOPK,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            warps=warps,
            stages=stages,
            grid_tokens=grid_tokens,
            out_key=("gu", 0, num_valid, w13.shape[1], DTYPE),
            naive=naive,
        )

    # Warm once to allocate scratch
    gate_up = stage_gemm1()
    _sync()

    def stage_layout_only():
        up, gate = gate_up.chunk(2, dim=-1)
        return torch.cat((gate, up), dim=-1)

    laid = stage_layout_only()
    _sync()

    def stage_silu_only():
        return silu_and_mul(laid)

    def stage_silu_up_gate():
        return silu_and_mul_up_gate(gate_up)

    inter = stage_silu_up_gate()
    tw_flat = topk_weights.reshape(-1).contiguous()
    _sync()

    def stage_gemm2():
        return _gemm(
            inter,
            w2,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_post=num_post,
            num_valid=num_valid,
            row_div=1,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            warps=warps,
            stages=stages,
            grid_tokens=grid_tokens,
            out_key=("dn", 0, num_valid, w2.shape[1], DTYPE),
            naive=naive,
            topk_w=tw_flat,
        )

    down = stage_gemm2()
    _sync()

    def stage_sum():
        return _moe_sum(down, M, TOPK)

    # Warmup each stage
    for fn in (
        stage_router,
        stage_prepare,
        stage_gemm1,
        stage_layout_only,
        stage_silu_only,
        stage_silu_up_gate,
        stage_gemm2,
        stage_sum,
    ):
        for _ in range(WARMUP):
            fn()
    _sync()

    stages_ms = {
        "router": timed_ms(stage_router, iters=ITERS),
        "prepare": timed_ms(stage_prepare, iters=ITERS),
        "gemm1": timed_ms(stage_gemm1, iters=ITERS),
        "layout_chunk_cat": timed_ms(stage_layout_only, iters=ITERS),
        "silu_only": timed_ms(stage_silu_only, iters=ITERS),
        "silu_up_gate": timed_ms(stage_silu_up_gate, iters=ITERS),
        "gemm2": timed_ms(stage_gemm2, iters=ITERS),
        "moe_sum": timed_ms(stage_sum, iters=ITERS),
    }

    # Full forward end-to-end for one layer
    from src.layers.moe.triton_moe import triton_moe_forward

    def full_experts():
        return triton_moe_forward(
            hidden_states=hidden,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13_weight=w13,
            w2_weight=w2,
        )

    for _ in range(WARMUP):
        full_experts()
    stages_ms["experts_e2e"] = timed_ms(full_experts, iters=ITERS)
    stages_ms["moe_block_est"] = stages_ms["router"] + stages_ms["experts_e2e"]

    # Scale to ~24 MoE layers (Qwen1.5-MoE has 24 layers, all MoE)
    n_layers = 24
    return {
        "M": M,
        "config": {
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
            "warps": warps,
            "stages": stages,
        },
        "pad": stats,
        "ms": stages_ms,
        "projected_24L_ms": {
            k: v * n_layers for k, v in stages_ms.items()
        },
    }


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYS)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    torch.cuda.set_device(0)
    print(
        f"GPU={PHYS} H={H} I={I} E={E} topk={TOPK} lengths={LENGTHS} "
        f"warmup={WARMUP} iters={ITERS}"
    )

    rows = []
    for M in LENGTHS:
        r = bench_one(M)
        rows.append(r)
        ms = r["ms"]
        pad = r["pad"]
        print(
            f"M={M:4d} naive={pad['naive']} blk={r['config']['block_m']} "
            f"pad={pad['pad_ratio']:.2f}x "
            f"router={ms['router']:.3f} prep={ms['prepare']:.3f} "
            f"g1={ms['gemm1']:.3f} silu={ms['silu_up_gate']:.3f} "
            f"(layout={ms['layout_chunk_cat']:.3f}) "
            f"g2={ms['gemm2']:.3f} sum={ms['moe_sum']:.3f} "
            f"e2e={ms['experts_e2e']:.3f} "
            f"| 24L≈{r['projected_24L_ms']['moe_block_est']:.1f}ms"
        )

    out_dir = ROOT / "bench" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "gpu": PHYS,
        "model_dims": {"H": H, "I": I, "E": E, "topk": TOPK},
        "rows": rows,
    }
    jp = out_dir / f"moe_layer_breakdown_{stamp}.json"
    jp.write_text(json.dumps(payload, indent=2))

    # Markdown table
    lines = [
        "# MoE single-layer stage breakdown (v7 Triton)",
        "",
        f"- GPU: `{PHYS}`",
        f"- Dims: H={H}, I={I}, E={E}, topk={TOPK}, bf16",
        f"- Warmup/iters: {WARMUP}/{ITERS}",
        f"- Projected TTFT MoE portion ≈ 24 × (router + experts_e2e)",
        "",
        "| M | naive | blk_m | pad× | router | prepare | GEMM1 | silu | layout | GEMM2 | sum | e2e | 24L MoE≈ |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        ms = r["ms"]
        pad = r["pad"]
        lines.append(
            f"| {r['M']} | {pad['naive']} | {r['config']['block_m']} | "
            f"{pad['pad_ratio']:.2f} | "
            f"{ms['router']:.3f} | {ms['prepare']:.3f} | {ms['gemm1']:.3f} | "
            f"{ms['silu_up_gate']:.3f} | {ms['layout_chunk_cat']:.3f} | "
            f"{ms['gemm2']:.3f} | {ms['moe_sum']:.3f} | {ms['experts_e2e']:.3f} | "
            f"{r['projected_24L_ms']['moe_block_est']:.1f} |"
        )
    # Percentage of e2e for M in short-prompt regime
    lines += ["", "## Share of experts_e2e (selected M)", ""]
    for r in rows:
        if r["M"] not in (1, 8, 32, 128):
            continue
        ms = r["ms"]
        e2e = ms["experts_e2e"]
        parts = [
            ("prepare", ms["prepare"]),
            ("gemm1", ms["gemm1"]),
            ("silu_up_gate", ms["silu_up_gate"]),
            ("gemm2", ms["gemm2"]),
            ("moe_sum", ms["moe_sum"]),
        ]
        lines.append(f"### M={r['M']} (e2e={e2e:.3f} ms)")
        for name, v in parts:
            lines.append(f"- {name}: {v:.3f} ms ({100*v/max(e2e,1e-9):.1f}%)")
        lines.append("")

    mp = out_dir / f"moe_layer_breakdown_{stamp}.md"
    mp.write_text("\n".join(lines) + "\n")
    print(f"Wrote {jp}")
    print(f"Wrote {mp}")


if __name__ == "__main__":
    main()
