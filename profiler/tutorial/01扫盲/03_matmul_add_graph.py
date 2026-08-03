import argparse
import math
import os

import torch

COMPILE_MODES = ["default", "max-autotune-no-cudagraphs"]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="对比 eager / torch.compile / 手动 CUDA Graph 三种方案的剖析结果")
    parser.add_argument("--size", type=int, default=64, help="矩阵尺寸 N")
    parser.add_argument("--mode", choices=["eager", "compile", "compile_graph"],
                        default="compile_graph", help="对比方案")
    parser.add_argument("--cmode", choices=COMPILE_MODES,
                        default="max-autotune-no-cudagraphs",
                        help="torch.compile 模式")
    parser.add_argument("--steps", type=int, default=20, help="正式记录步数")
    parser.add_argument("--warm", type=int, default=10, help="剖析前预热步数")
    parser.add_argument("--trace_dir", default="./traces/01_matmul_add_graph",
                        help="trace 输出目录")
    return parser.parse_args()


def check_correctness(out, ref, N):
    max_diff = (out.float() - ref.float()).abs().max().item()
    assert max_diff < 0.1 * math.sqrt(N), f"结果不一致, max diff={max_diff:.3f}"
    print(f"[正确性] 最大偏差 {max_diff:.3f}")


def capture_cuda_graph(cf, xs, ws, bs, ref, N):
    """把编译后的函数捕获进 CUDA Graph, 返回图对象。

    1. 预热: 先跑 3 次让 torch.compile 完成编译, 否则编译过程会被录进图里;
    2. 捕获: with torch.cuda.graph 中的 kernel 只"记录"不执行, 此时 y 无值;
    3. replay: 捕获后必须真正执行一次, y 才有值, 顺带完成正确性校验。
    """
    for _ in range(3):
        cf(xs, ws, bs)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = cf(xs, ws, bs)
    torch.cuda.synchronize()

    graph.replay()
    torch.cuda.synchronize()
    check_correctness(y, ref, N)
    return graph


def build_step(args, fn, x, w, b, ref):
    """按 --mode 构建"单步执行函数", 并完成该模式特有的准备工作。

    step() 每次被调用都会在 trace 中打上对应标签, 剖析器据此区分三种方案:
    - eager:        直接执行原始算子;
    - compile:      执行 torch.compile 融合后的 kernel;
    - compile_graph: 手动把融合 kernel 捕获进 CUDA Graph, 之后只 replay。
    """
    if args.mode == "eager":
        def step():
            with torch.profiler.record_function("eager_mm_add"):
                return fn(x, w, b)
        return step

    # 编译: 编译器把 matmul + add 融合成一个 kernel
    cf = torch.compile(fn, mode=args.cmode)
    check_correctness(cf(x, w, b), ref, args.size)

    if args.mode == "compile":
        def step():
            with torch.profiler.record_function("compile_fused"):
                return cf(x, w, b)
        return step

    xs, ws, bs = x.clone(), w.clone(), b.clone()
    graph = capture_cuda_graph(cf, xs, ws, bs, ref, args.size)

    def step():
        with torch.profiler.record_function("compile_graph"):
            return graph.replay()
    return step


def profile_and_export(step, trace_dir, tag, steps):
    """运行剖析器记录 step 的耗时, 并导出汇总表与 Chrome trace。"""
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=steps, repeat=1)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,    # 记录 CPU 侧的调度/算子下发
            torch.profiler.ProfilerActivity.CUDA,   # 记录 GPU 侧的 kernel 执行
        ],
        schedule=schedule,
        record_shapes=False,   # 不需要形状信息, 关闭以降低剖析开销
        profile_memory=False,  # 不分析显存, 只关注耗时
        with_stack=False,      # 不需要调用栈, 关闭以避免记录 Python 帧开销
    ) as prof:
        for _ in range(steps + 2):
            step()
            prof.step()

    torch.cuda.synchronize()

    os.makedirs(trace_dir, exist_ok=True)
    table_path = os.path.join(trace_dir, f"{tag}.txt")
    trace_path = os.path.join(trace_dir, f"{tag}.json")

    prof.export_chrome_trace(trace_path)
    print(f"Chrome trace → {trace_path}")
    with open(table_path, "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


def main():
    args = parse_arguments()
    N = args.size

    # 构造随机输入 (bf16, bias 为 1D), 与 README 中的算式保持一致
    x = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, device="cuda", dtype=torch.bfloat16)

    def fn(x, w, b):
        return torch.add(torch.matmul(x, w), b)

    # 以 eager 结果为基准, 之后每个方案都要与它做正确性比对
    ref = fn(x, w, b)

    # 构建 step 函数 (编译 / 图捕获等准备工作都在此完成)
    step = build_step(args, fn, x, w, b, ref)

    # 预热: 在剖析器外再跑 warm 次, 让 GPU 进入稳定频率, 避免污染剖析数据
    for _ in range(args.warm):
        step()
    torch.cuda.synchronize()

    # 正式剖析并导出结果, 文件名以 "尺寸_方案" 区分
    profile_and_export(step, args.trace_dir, f"{N}_{args.mode}", args.steps)


if __name__ == "__main__":
    main()
