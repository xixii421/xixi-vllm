""" 02融合: 手写 Triton 融合 kernel (matmul+bias) + 手动 CUDA Graph。

对比三种方案在 64/4096 下的开销:
  eager     torch.matmul + torch.add (cuBLAS, 两个 kernel)
  triton    手写 Triton 融合 kernel (一个 kernel, matmul+bias 合一)
  graph     在上者基础上手动捕获 CUDA Graph, 循环外 replay

导出 Chrome trace 供 Perfetto 查看: https://ui.perfetto.dev/
"""

import argparse
import math
import os

import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 64, 64  # 融合 kernel 的 tile 尺寸 (64/4096 均可整除)


@triton.jit
def _mm_add_kernel(X, W, B, Y, M, N, K,
                   sx_m, sx_k, sw_k, sw_n, sb_n, sy_m, sy_n,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # 每个 program 负责 (BM x BN) 的输出 tile
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)

    xp = X + om[:, None] * sx_m + ok[None, :] * sx_k
    wp = W + ok[:, None] * sw_k + on[None, :] * sw_n

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(xp)
        c = tl.load(wp)
        acc += tl.dot(a, c, out_dtype=tl.float32)
        xp += BK * sx_k
        wp += BK * sw_k

    bias = tl.load(B + on * sb_n)
    tl.store(Y + om[:, None] * sy_m + on[None, :] * sy_n,
             (acc + bias[None, :]).to(Y.dtype.element_ty))


def fused_mm_add(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """手写 Triton 融合: y = x @ w + b (单 kernel)。"""
    M, K = x.shape
    N = w.shape[1]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    _mm_add_kernel[(M // BM, N // BN)](
        x, w, b, y, M, N, K,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1), b.stride(0),
        y.stride(0), y.stride(1),
        BM, BN, BK,
    )
    return y


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=64, help="矩阵尺寸 N")
    p.add_argument("--mode", choices=["eager", "triton", "graph"],
                   default="eager", help="eager=cuBLAS, triton=手写融合, graph=triton+CUDA Graph")
    p.add_argument("--steps", type=int, default=20, help="正式记录步数")
    p.add_argument("--warm", type=int, default=10, help="剖析前预热步数")
    p.add_argument("--trace_dir", default="./traces/02_fused_mm_add", help="trace 输出目录")
    return p.parse_args()


def main():
    args = parse_arguments()
    N = args.size
    x = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, device="cuda", dtype=torch.bfloat16)  # 1D bias, 广播到每行

    # 预校验: bf16 舍入路径不同 (Triton fp32 一次累加 vs cublas 中途舍入),
    # 误差随 K 累积, 用 0.1*sqrt(N) 作为合理阈值
    ref = torch.add(torch.matmul(x, w), b)
    out = fused_mm_add(x, w, b)
    max_diff = (out.float() - ref.float()).abs().max().item()
    assert max_diff < 0.1 * math.sqrt(N), f"Triton kernel 结果不一致, max diff={max_diff:.3f}"
    print(f"[正确性] Triton 与 eager 最大偏差 {max_diff:.3f}")

    # 按 mode 构造 step: eager / triton / triton+CUDA Graph
    if args.mode == "eager":
        def step():
            with torch.profiler.record_function("eager_mm_add"):
                return torch.add(torch.matmul(x, w), b)
    elif args.mode == "triton":
        def step():
            with torch.profiler.record_function("triton_fused"):
                return fused_mm_add(x, w, b)
    else:
        # CUDA Graph: 输入必须驻留静态 buffer, 捕获一次, 之后只 replay
        x_s, w_s, b_s = x.clone(), w.clone(), b.clone()
        for _ in range(3):
            fused_mm_add(x_s, w_s, b_s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused_mm_add(x_s, w_s, b_s)
        torch.cuda.synchronize()

        def step():
            with torch.profiler.record_function("triton_graph"):
                return graph.replay()

    # 预热: 在 profiler 外完成编译/图捕获, 避免污染数据
    for _ in range(args.warm):
        step()
    torch.cuda.synchronize()

    os.makedirs(args.trace_dir, exist_ok=True)
    tag = f"{N}_{args.mode}"
    table_path = os.path.join(args.trace_dir, f"{tag}.txt")
    trace_path = os.path.join(args.trace_dir, f"{tag}.json")

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=args.steps, repeat=1)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(args.steps + 2):
            step()
            prof.step()

    torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)
    print(f"Chrome trace → {trace_path}")
    with open(table_path, "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


if __name__ == "__main__":
    main()
