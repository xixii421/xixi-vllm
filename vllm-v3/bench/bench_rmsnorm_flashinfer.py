"""RMSNorm 融合对比：v3 现状（x+residual 后 F.rms_norm） vs FlashInfer fused_add_rmsnorm。

对照 docs/05-算子融合.md：v3 的残差加法与 RMSNorm 是两个 kernel（residual 写回 HBM 再读回），
FlashInfer fused_add_rmsnorm 是单 kernel 把 add 融进 norm。这里在多个规模下测二者差距，
并对比两种 custom-op 包装方案（clone 版 vs mutating 版）的开销。

结论（GPU1 RTX4070S）：
- 原生 fused 比现状快约 1.8x（hidden=1024）；但直接调用会让 torch.compile
  reduce-overhead 的 cudagraph 捕获产生空图（prefill 性能暴跌），不可直接用。
- clone 版（non-mutating custom op）：为保持纯函数语义多一次 x/residual 拷贝，
  反而比现状慢 25%，融合收益被拷贝开销吞掉。
- mutating 版（mutates_args=('x','residual')）：torch.compile 能正确推断别名，
  cudagraph 正常，性能与原生 fused 相当，这是 v3 采用的方式。

规模覆盖 Qwen3-0.6B（hidden=1024）的三种典型场景：
- prefill 大批量: (1, 4096) / (8, 512)
- 中等: (32, 128)
- decode 小批量: (64, 1) / (256, 1)
同样测 hidden=4096 看大模型趋势。

用法:
    python bench/bench_rmsnorm_flashinfer.py
    python bench/bench_rmsnorm_flashinfer.py --iters 500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from flashinfer.norm import fused_add_rmsnorm

BENCH = Path(__file__).resolve().parent


def bench_fn(fn, iters: int, warmup: int = 30) -> float:
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
    """相对误差容差（bf16 下 FlashInfer rcp.approx 近似约有 1% 相对误差）。"""
    return bool(((a - b).abs() / (a.abs() + 1e-3)).max().item() < tol)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--iters", type=int, default=300)
    args = p.parse_args()
    torch.cuda.set_device(args.device)

    g = torch.Generator(device="cuda").manual_seed(0)
    lines: list[str] = []
    print("== RMSNorm 融合对比（CUDA event 计时，bf16）==\n")

    header = "| 形状 | v3现状 add+F (ms) | FI fused custom-op (ms) | 加速比(vs现状) |"
    print(header)
    lines.append(header)
    lines.append("|---|---|---|---|---|")

    shapes = [
        (1, 4096, 1024),   # prefill 长序列
        (8, 512, 1024),    # prefill 中等
        (32, 128, 1024),   # 合批中等
        (64, 1, 1024),     # decode
        (256, 1, 1024),    # decode 大批
        (1, 4096, 4096),   # 大模型 prefill
        (32, 128, 4096),   # 大模型 中等
    ]
    for (b, s, h) in shapes:
        n = b * s
        x = torch.randn(n, h, dtype=torch.bfloat16, device="cuda", generator=g)
        residual = torch.randn(n, h, dtype=torch.bfloat16, device="cuda", generator=g)
        weight = torch.randn(h, dtype=torch.bfloat16, device="cuda", generator=g) * 0.02 + 1

        # v3 现状：residual = x + residual; F.rms_norm(residual)
        def v3():
            r = x + residual
            return F.rms_norm(r, (h,), weight, 1e-6), r

        # custom-op 包装版：当前 v3 采用 mutating 语义（mutates_args=('x','residual')），
        # 不 clone，cudagraph 兼容。flashinfer 底层仍是一次 inplace 融合。
        fi_in = x.clone()
        fi_res = residual.clone()

        def custom():
            fused_add_rmsnorm(fi_in, fi_res, weight, 1e-6, enable_pdl=False)

        # 正确性
        o3, r3 = v3()
        custom()
        assert close(o3, fi_in, 1e-2), f"custom-op 与 v3 输出不一致 shape={b}x{s}x{h}"
        assert close(r3, fi_res, 1e-2)

        t3 = bench_fn(v3, args.iters)
        tcustom = bench_fn(custom, args.iters)

        row = (
            f"| {b}x{s}x{h} | {t3:.4f} | {tcustom:.4f} | **{t3/tcustom:.2f}x** |"
        )
        print(row)
        lines.append(row)

    out = BENCH / "out"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "bench_rmsnorm_flashinfer.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"\n→ {path}")


if __name__ == "__main__":
    main()
