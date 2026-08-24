"""算子级微基准：v2 vs v3（RMSNorm / RoPE / SiLU+Mul / MLP）。

对照 docs/02-实测证据.md 第一部分，隔离测量单个算子的 v2 / v3 实现差异。
不含权重加载与引擎调度，只测算子本身，CUDA event 计时，warmup 后取均值。

被测算子（忠实复刻源码）：
- RMSNorm   v2: pow/mean/rsqrt 三段式 | v3: F.rms_norm 融合核
- RoPE      v2: index_select + chunk + rotate | v3: FlashInfer apply_rope
- SiLU+Mul  v2: F.silu * up | v3: FlashInfer silu_and_mul
- MLP       v2: gate/up 双 GEMM + cat | v3: gate_up 融合 GEMM

形状对齐 Qwen3-0.6B：hidden=1024, intermediate=3072, heads=16, head_dim=128。

用法:
    python bench/bench_ops.py
    python bench/bench_ops.py --iters 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from flashinfer import apply_rope_with_cos_sin_cache
from flashinfer.activation import silu_and_mul

BENCH = Path(__file__).resolve().parent

HIDDEN = 1024
INTER = 3072
NHEADS = 16
HEAD_DIM = 128
MAX_POS = 32768


# --------------------------------------------------------------------------- #
# 计时工具
# --------------------------------------------------------------------------- #
def bench_fn(fn, iters: int, warmup: int = 25) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def close(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-2) -> bool:
    return bool((a - b).abs().max().item() < tol)


# --------------------------------------------------------------------------- #
# 各算子 v2 / v3 实现
# --------------------------------------------------------------------------- #
# RMSNorm
def rmsnorm_v2(x, weight, eps=1e-6):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * weight).to(x.dtype)


def rmsnorm_v3(x, weight, eps=1e-6):
    return F.rms_norm(x, (weight.shape[0],), weight, eps)


# RoPE
def build_cos_sin(head_size: int, base: float = 1e6) -> torch.Tensor:
    inv = 1.0 / (base ** (torch.arange(0, head_size, 2, dtype=torch.float) / head_size))
    t = torch.arange(MAX_POS, dtype=torch.float)
    freqs = torch.outer(t, inv)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).cuda()  # [MAX_POS, head_size] fp32


def rope_v2(x: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor, head_size: int):
    flat = positions.reshape(-1)
    cs = cos_sin.index_select(0, flat).to(x.dtype)
    cos, sin = cs.chunk(2, dim=-1)
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    shape = x.shape
    x = x.reshape(flat.numel(), -1, head_size)
    x1, x2 = x.chunk(2, dim=-1)
    rot = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return rot.reshape(shape)


def rope_v3(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor, head_size: int):
    qf = q.reshape(q.shape[0], -1).contiguous()
    kf = k.reshape(k.shape[0], -1).contiguous()
    qo, ko = apply_rope_with_cos_sin_cache(positions, qf, kf, head_size, cos_sin, is_neox=True)
    return qo.view_as(q), ko.view_as(k)


# SiLU+Mul
def silu_mul_v2(x: torch.Tensor):
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


def silu_mul_v3(x: torch.Tensor):
    return silu_and_mul(x)


# --------------------------------------------------------------------------- #
# 测试入口
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()
    torch.cuda.set_device(args.device)

    g = torch.Generator(device="cuda").manual_seed(0)
    lines: list[str] = []
    print("== 算子级 v2 vs v3（Qwen3-0.6B 形状，bf16，CUDA event）==\n")

    # RMSNorm
    weight = torch.ones(HIDDEN, dtype=torch.bfloat16, device="cuda")
    print("## RMSNorm")
    hdr = "| 形状 | v2 (ms) | v3 (ms) | 加速比 |"
    print(hdr)
    lines.append(hdr)
    for shape in [(1, 4096, HIDDEN), (8, 256, HIDDEN), (32, 1, HIDDEN)]:
        x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda", generator=g)
        assert close(rmsnorm_v2(x, weight), rmsnorm_v3(x, weight))
        t2 = bench_fn(lambda: rmsnorm_v2(x, weight), args.iters)
        t3 = bench_fn(lambda: rmsnorm_v3(x, weight), args.iters)
        row = f"| {shape[0]}x{shape[1]}x{shape[2]} | {t2:.3f} | {t3:.3f} | **{t2/t3:.2f}x** |"
        print(row)
        lines.append(row)

    # RoPE
    cos_sin = build_cos_sin(HEAD_DIM)
    print("\n## RoPE")
    hdr = "| 形状 | v2 (ms) | v3 (ms) | 加速比 |"
    print(hdr)
    lines.append(hdr)
    NKV = 8
    for n in [4096, 256, 32]:
        q = torch.randn(n, NHEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda", generator=g)
        k = torch.randn(n, NKV, HEAD_DIM, dtype=torch.bfloat16, device="cuda", generator=g)
        pos = torch.randint(0, MAX_POS, (n,), dtype=torch.int32, device="cuda", generator=g)
        q2, k2 = rope_v2(q, pos, cos_sin, HEAD_DIM), rope_v2(k, pos, cos_sin, HEAD_DIM)
        q3, k3 = rope_v3(q, k, pos, cos_sin, HEAD_DIM)
        assert close(q2, q3, 0.05) and close(k2, k3, 0.05)
        t2 = bench_fn(lambda: (rope_v2(q, pos, cos_sin, HEAD_DIM), rope_v2(k, pos, cos_sin, HEAD_DIM)), args.iters)
        t3 = bench_fn(lambda: rope_v3(q, k, pos, cos_sin, HEAD_DIM), args.iters)
        row = f"| {n}x{NHEADS}x{HEAD_DIM} | {t2:.3f} | {t3:.3f} | **{t2/t3:.2f}x** |"
        print(row)
        lines.append(row)

    # SiLU+Mul
    print("\n## SiLU+Mul")
    hdr = "| 形状 | v2 (ms) | v3 (ms) | 加速比 |"
    print(hdr)
    lines.append(hdr)
    for shape in [(1, 4096, INTER * 2), (8, 256, INTER * 2), (32, 1, INTER * 2)]:
        x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda", generator=g)
        assert close(silu_mul_v2(x), silu_mul_v3(x), 0.5)
        t2 = bench_fn(lambda: silu_mul_v2(x), args.iters)
        t3 = bench_fn(lambda: silu_mul_v3(x), args.iters)
        row = f"| {shape[0]}x{shape[1]}x{shape[2]} | {t2:.3f} | {t3:.3f} | **{t2/t3:.2f}x** |"
        print(row)
        lines.append(row)

    # MLP
    print("\n## MLP（gate/up 投影 + act + down）")
    gate = torch.randn(INTER, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=g) * 0.02
    up = torch.randn(INTER, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=g) * 0.02
    gate_up = torch.cat([gate, up], dim=0).contiguous()
    down = torch.randn(HIDDEN, INTER, dtype=torch.bfloat16, device="cuda", generator=g) * 0.02

    def mlp_v2(x):
        g = F.linear(x, gate)
        u = F.linear(x, up)
        return F.linear(silu_mul_v2(torch.cat([g, u], dim=-1)), down)

    def mlp_v3(x):
        gu = F.linear(x, gate_up)
        return F.linear(silu_mul_v3(gu), down)

    hdr = "| 形状 | v2 (ms) | v3 (ms) | 加速比 |"
    print(hdr)
    lines.append(hdr)
    for shape in [(1, 4096, HIDDEN), (8, 256, HIDDEN), (32, 1, HIDDEN)]:
        x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda", generator=g)
        assert close(mlp_v2(x), mlp_v3(x), 0.5)
        t2 = bench_fn(lambda: mlp_v2(x), args.iters)
        t3 = bench_fn(lambda: mlp_v3(x), args.iters)
        row = f"| {shape[0]}x{shape[1]}x{shape[2]} | {t2:.3f} | {t3:.3f} | **{t2/t3:.2f}x** |"
        print(row)
        lines.append(row)

    out = BENCH / "out"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "bench_ops.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"\n→ {path}")


if __name__ == "__main__":
    main()
