""" 遍历 compile 各模式，在指定矩阵尺寸下各跑 N 次测量稳态延迟。

用法:
  python 02_bench_modes.py --size 64  --n 100
  python 02_bench_modes.py --size 4096 --n 100
"""

import argparse
import statistics
import time

import torch

MODES = ["none", "default", "reduce-overhead",
         "max-autotune", "max-autotune-no-cudagraphs"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=64, help="矩阵尺寸 N")
    p.add_argument("--n", type=int, default=100, help="每种模式测量次数")
    p.add_argument("--warm", type=int, default=20, help="每种模式预热次数")
    args = p.parse_args()
    assert args.n >= 10, "--n 太小，统计指标无意义"

    N = args.size
    x = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, device="cuda", dtype=torch.bfloat16)   # 1D bias, 与 03 一致

    def fn(x, w, b):
        return torch.add(torch.matmul(x, w), b)

    def bench(f):
        """预热触发编译/图捕获, 再正式测量, 返回 (编译耗时, 中位, 平均, p95)。"""
        t0 = time.perf_counter()
        for _ in range(args.warm):
            f(x, w, b)
        torch.cuda.synchronize()
        compile_s = time.perf_counter() - t0
        ts = []
        for _ in range(args.n):
            t0 = time.perf_counter()
            f(x, w, b)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e6)
        ts.sort()
        return (compile_s,
                statistics.median(ts),
                sum(ts) / len(ts),
                ts[int(len(ts) * 0.95) - 1])

    print(f"\n矩阵 {N}x{N}  ×  {args.n} 次测量   (中位/平均/p95 均为微秒)")
    print(f"{'mode':<28s} {'编译耗时':>9s} {'中位(us)':>10s} "
          f"{'平均(us)':>10s} {'p95(us)':>10s}  {'相对none':>8s}")
    print("-" * 84)

    # 先独立测 eager 基准, 不依赖 MODES 循环顺序
    base = bench(fn)[1]
    for mode in MODES:
        f = fn if mode == "none" else torch.compile(fn, mode=mode)
        compile_s, med, avg, p95 = bench(f)
        print(f"{mode:<28s} {compile_s:>8.2f}s {med:>9.1f} "
              f"{avg:>10.1f} {p95:>10.1f}  {med / base:>7.2f}x")


if __name__ == "__main__":
    main()
