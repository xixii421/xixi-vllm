""" 剖析 matmul + add 算子。

通过 --size / --dtype / --mode / --steps / --warm 组合，
对比 eager 与 torch.compile 不同模式下的 GPU 耗时，导出 Chrome trace 供 Perfetto 查看。
Perfetto: https://ui.perfetto.dev/
"""

import argparse
import os

import torch

COMPILE_MODES = ["none", "default", "reduce-overhead",
                 "max-autotune", "max-autotune-no-cudagraphs"]


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=64, help="矩阵尺寸 N")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--mode", choices=COMPILE_MODES, default="none",
                   help="编译模式, none 表示 eager 模式")
    p.add_argument("--steps", type=int, default=20,
                   help="正式记录的步数 (active 阶段)")
    p.add_argument("--warm", type=int, default=10,
                   help="剖析前预热步数，用于让 torch.compile 完成编译/图捕获")
    p.add_argument("--trace_dir", default="./traces/01_matmul_add",
                   help="trace 输出目录")
    return p.parse_args()


def main():
    args = parse_arguments()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    x = torch.randn(args.size, args.size, device="cuda", dtype=dtype)
    w = torch.randn(args.size, args.size, device="cuda", dtype=dtype)
    b = torch.randn(args.size, args.size, device="cuda", dtype=dtype)

    # 定义乘加算子
    def fn(x, w, b):
        return torch.add(torch.matmul(x, w), b)

    # 按模式包装算子
    if args.mode != "none":
        fn = torch.compile(fn, mode=args.mode)

    def step():
        # record_function 给 trace 里的算子打上易读标记，便于按阶段过滤
        with torch.profiler.record_function("matmul_add"):
            return fn(x, w, b)

    # 预热：在进入剖析器之前完成编译/图捕获，避免污染 profile 数据
    for _ in range(args.warm):
        step()
    torch.cuda.synchronize()

    os.makedirs(args.trace_dir, exist_ok=True)
    tag = f"{args.size}_{args.dtype}_{args.mode}"
    table_path = os.path.join(args.trace_dir, f"{tag}.txt")
    trace_path = os.path.join(args.trace_dir, f"{tag}.json")

    # schedule: 前 2 步不记录，随后连续记录 args.steps 步
    schedule = torch.profiler.schedule(
        wait=1, warmup=1, active=args.steps, repeat=1)

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

    # Chrome trace 格式，拖进 https://ui.perfetto.dev 即可查看
    prof.export_chrome_trace(trace_path)
    print(f"Chrome trace → {trace_path}")

    # 汇总表：按 CUDA 时间排序的算子耗时 Top 15
    with open(table_path, "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


if __name__ == "__main__":
    main()
