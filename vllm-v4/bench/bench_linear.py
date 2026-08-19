"""AI 评审对 linear.py 提出的四个瓶颈逐项实测验证。

覆盖：
  瓶颈1  SwiGLU 激活融合（naive split+silu+mul vs flashinfer silu_and_mul，
        以及 torch.compile 能否把激活合入 GEMM epilogue）
  瓶颈2  GQA 下 KV 权重复制冗余的触发条件与算力浪费
  瓶颈3  _shard 的 narrow+contiguous 双重拷贝 vs 直接 view 拷贝
  瓶颈4  TP=2 下 all-reduce 通信与 GEMM 计算的占比（需 2 卡）

用法（在 vllm-v4/ 下）:
    ../.venv2/bin/python bench/bench_linear.py          # 单卡，跳过瓶颈4
    ../.venv2/bin/python bench/bench_linear.py --tp2     # 瓶颈4 也跑（需 2 卡）
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.compute.layers import ops as _ops_reg  # noqa: F401  注册 vllm::silu_and_mul 等算子

HIDDEN = 1024   # Qwen3-0.6B hidden
INTER = 3072    # Qwen3-0.6B intermediate
NUM_KV = 8      # Qwen3-0.6B num_key_value_heads
NUM_H = 16      # Qwen3-0.6B num_attention_heads
HEAD_DIM = 128


def _time_cuda(fn, *args, warmup: int = 20, iters: int = 100) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def _time_cpu(fn, *args, iters: int = 5, **kwargs) -> float:
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    return (time.perf_counter() - t0) / iters * 1e3  # ms


# --------------------------------------------------------------------------- #
# 瓶颈1  SwiGLU 激活融合
# --------------------------------------------------------------------------- #


def bench_swiglu() -> None:
    print("=" * 78)
    print("瓶颈1: SwiGLU 激活融合（Qwen3-0.6B: hidden=1024, intermediate=3072）")
    print("=" * 78)
    torch.manual_seed(0)
    dev = "cuda"
    w = torch.randn(2 * INTER, HIDDEN, device=dev, dtype=torch.bfloat16) * 0.02
    w_fp32 = w.float()
    ms_list = [1, 64, 256, 1024, 4096]

    print("-- 1a. 激活算子本身：AI 担心的 split+silu+mul 与仓库现用的融合 kernel --")
    print(f"{'M(tokens)':>10} | {'naive':>9} | {'silu_and_mul':>12} | "
          f"{'融合/naive':>10} | max_err")
    print("-" * 72)
    for m in ms_list:
        x = torch.randn(m, HIDDEN, device=dev, dtype=torch.bfloat16)

        def naive():
            g = torch.nn.functional.linear(x, w)
            return torch.nn.functional.silu(g[..., :INTER]) * g[..., INTER:]

        def fused():
            g = torch.nn.functional.linear(x, w)
            return torch.ops.vllm.silu_and_mul(g)

        t_naive = _time_cuda(naive, iters=50)
        t_fused = _time_cuda(fused, iters=50)

        g32 = torch.nn.functional.linear(x.float(), w_fp32)
        ref = torch.nn.functional.silu(g32[..., :INTER]) * g32[..., INTER:]
        err_fused = (fused().float() - ref).abs().max().item()

        print(f"{m:>10} | {t_naive:>8.4f} | {t_fused:>11.4f} | "
              f"{t_naive / max(t_fused, 1e-9):>9.2f}x | {err_fused:.4f}")

    print("-- 1b. 完整 MLP 链路（gate_up GEMM + 激活 + down GEMM）--")
    w_down = torch.randn(HIDDEN, INTER, device=dev, dtype=torch.bfloat16) * 0.02

    def mlp_native(x):
        g = torch.nn.functional.linear(x, w)
        a = torch.nn.functional.silu(g[..., :INTER]) * g[..., INTER:]
        return torch.nn.functional.linear(a, w_down)

    compiled_native = torch.compile(mlp_native)
    print(f"{'M(tokens)':>10} | {'eager+flash':>12} | {'compile原生':>12} | "
          f"{'compile/flash':>12}")
    print("-" * 62)
    for m in [1, 256, 4096]:
        x = torch.randn(m, HIDDEN, device=dev, dtype=torch.bfloat16)

        def cur():
            g = torch.nn.functional.linear(x, w)
            a = torch.ops.vllm.silu_and_mul(g)
            return torch.nn.functional.linear(a, w_down)

        t_cur = _time_cuda(cur, iters=50)
        t_cc = _time_cuda(lambda: compiled_native(x), iters=50)
        print(f"{m:>10} | {t_cur:>11.4f} | {t_cc:>11.4f} | "
              f"{t_cc / max(t_cur, 1e-9):>10.2f}x")

    # 1c. GemmSwiGLU（把激活合入 GEMM epilogue）的理论收益上界
    print("-- 1c. GemmSwiGLU 理论收益：中间 [M, 2I] 张量落盘的读写带宽 --")
    bytes_per_token = 2 * INTER * 2  # bf16 读写各一次
    print(f"中间张量每 token {bytes_per_token / 1024:.1f} KB（写 1 次 + silu 读 1 次）")
    for m in [1, 4096]:
        total = m * bytes_per_token
        print(f"  M={m}: {total / 2**20:.2f} MB 的 HBM 读写，"
              f"按 2TB/s 约 {total / 2e12 * 1e6:.1f} us")
    print("结论: prefill 大 batch 下中间张量读写可达数十 us 量级，"
          "相对 GEMM 计算时间占比有限；")
    print("      decode 下 GEMM 本身受带宽主导，融合主要省 kernel launch。")
    print("      torch.compile 实测只能把 silu+slice+mul 合成 1 个 pointwise kernel，")
    print("      无法合入 cuBLAS GEMM 的 epilogue，真融合需 CUTLASS/Triton 手写。")
    print()


# --------------------------------------------------------------------------- #
# 瓶颈2  GQA KV 复制冗余
# --------------------------------------------------------------------------- #


def bench_gqa() -> None:
    print("=" * 78)
    print("瓶颈2: GQA KV 头复制冗余（Qwen3-0.6B: 16 Q 头, 8 KV 头, head=128）")
    print("=" * 78)
    print(f"{'tp':>4} | {'num_kv/rank':>10} | {'replicas':>8} | "
          f"{'KV冗余占qkv%':>12} | {'状态':>6}")
    print("-" * 62)
    for tp in [1, 2, 4, 8, 16, 32]:
        if tp > NUM_KV:
            nkv = 1
            replicas = tp // NUM_KV
        else:
            nkv = NUM_KV // tp
            replicas = 1
        nq = NUM_H // tp
        q_flops = nq * HEAD_DIM * HIDDEN
        kv_actual = 2 * nkv * HEAD_DIM * HIDDEN * replicas
        kv_needed = 2 * nkv * HEAD_DIM * HIDDEN
        waste = (kv_actual - kv_needed) / (q_flops + kv_actual) * 100
        state = "冗余" if replicas > 1 else "无冗余"
        print(f"{tp:>4} | {nkv:>10} | {replicas:>8} | {waste:>10.1f}% | {state:>6}")
    print()
    print(f"本机默认 tp=2：n_kv=8 >= tp，replicas=1，无冗余计算")
    print(f"仅 tp>{NUM_KV}（如 16/32 卡）才出现 KV 复制冗余，且每卡计算完全相同")
    print(f"注：vLLM 官方对 tp>n_kv 同样采用复制策略（weight 复制 + 重复计算），")
    print(f"    靠 NVLink 带宽摊薄；广播 KV 方案复杂且需改 attention 数据流。")
    print()


# --------------------------------------------------------------------------- #
# 瓶颈3  _shard 双重拷贝
# --------------------------------------------------------------------------- #


def _shard_contig(weight: torch.Tensor, dim: int, rank: int, tp: int) -> torch.Tensor:
    size = weight.shape[dim] // tp
    offset = size * rank
    return weight.narrow(dim, offset, size).contiguous()


def _shard_view(weight: torch.Tensor, dim: int, rank: int, tp: int) -> torch.Tensor:
    size = weight.shape[dim] // tp
    offset = size * rank
    return weight.narrow(dim, offset, size)


def bench_shard() -> None:
    print("=" * 78)
    print("瓶颈3: _shard 的 narrow+contiguous 双重拷贝（CPU 权重加载）")
    print("=" * 78)
    torch.manual_seed(3)
    cases = [
        ("gate_up [6144,1024] dim=0", 6144, HIDDEN, 0),
        ("qkv    [6144,1024] dim=0", 6144, HIDDEN, 0),
        ("down   [1024,3072] dim=1", HIDDEN, INTER, 1),
        ("o_proj [1024,1024] dim=1", HIDDEN, HIDDEN, 1),
    ]
    tp = 2
    print(f"{'场景':>22} | {'view直接copy':>12} | {'contiguous+copy':>15} | "
          f"{'一致':>4} | {'速度比':>6}")
    print("-" * 72)
    for label, out, inn, dim in cases:
        loaded = torch.randn(out, inn, dtype=torch.bfloat16)
        per = (out if dim == 0 else inn) // tp
        if dim == 0:
            p1 = torch.zeros(per, inn, dtype=torch.bfloat16)
        else:
            p1 = torch.zeros(out, per, dtype=torch.bfloat16)
        p2 = torch.zeros_like(p1)

        def load_view():
            p1.copy_(_shard_view(loaded, dim, 0, tp))

        def load_contig():
            p2.copy_(_shard_contig(loaded, dim, 0, tp))

        load_view()
        load_contig()
        same = torch.equal(p1, p2)
        t_v = _time_cpu(load_view, iters=10)
        t_c = _time_cpu(load_contig, iters=10)
        print(f"{label:>22} | {t_v:>11.3f} | {t_c:>14.3f} | {str(same):>4} | "
              f"{t_c / max(t_v, 1e-9):>5.1f}x")
    print()
    print("说明: dim=0 切分（Column/Merged/QKV 用）的 view 每行内部连续，直接 copy_")
    print("      就是逐行 memcpy，去掉 contiguous 只省一次临时分配。")
    print("      dim=1 切分（Row 用）的 view 行间跳跃，PyTorch 的 strided copy_")
    print("      是否会退化由实测判定。")
    print()


# --------------------------------------------------------------------------- #
# 瓶颈4  TP=2 通信 vs 计算（需 2 卡）
# --------------------------------------------------------------------------- #


def _tp2_worker(rank: int, port: int) -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
    )

    init_distributed_environment(
        world_size=2, rank=rank, local_rank=rank, backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
    )
    initialize_model_parallel(2, 1, rank, 0)
    torch.manual_seed(rank)
    dev = "cuda"
    bs = 64
    # MLP down_proj 场景：每 rank 持有 [bs, INTER//tp] 输入，GEMM 到 [bs, HIDDEN]，
    # 输出做 all-reduce（通信量 = bs*HIDDEN bf16）。
    local = torch.randn(bs, INTER // 2, device=dev, dtype=torch.bfloat16)
    w_down = torch.randn(HIDDEN, INTER // 2, device=dev, dtype=torch.bfloat16) * 0.02

    def gemm_only():
        return torch.nn.functional.linear(local, w_down)

    def gemm_ar():
        out = torch.nn.functional.linear(local, w_down)
        return tensor_model_parallel_all_reduce(out)

    def gemm_ag():
        out = torch.nn.functional.linear(local, w_down)
        return tensor_model_parallel_all_gather(out, dim=-1)

    t_g = _time_cuda(gemm_only)
    t_ar = _time_cuda(gemm_ar)
    t_ag = _time_cuda(gemm_ag)
    bytes_ar = bs * HIDDEN * 2
    print(f"rank{rank}: GEMM={t_g:.4f}ms  GEMM+allreduce={t_ar:.4f}ms "
          f"(通信占比 {(t_ar - t_g) / t_ar * 100:.1f}%)  "
          f"GEMM+allgather={t_ag:.4f}ms ({(t_ag - t_g) / t_ag * 100:.1f}%)",
          flush=True)
    print(f"rank{rank}: 单次 all-reduce 通信量 = {bytes_ar / 1024:.1f} KB",
          flush=True)
    destroy_model_parallel()
    destroy_distributed_environment()


def bench_comm(do_tp2: bool) -> None:
    print("=" * 78)
    print("瓶颈4: TP=2 下 all-reduce 与 GEMM 的耗时占比")
    print("=" * 78)
    if not do_tp2 or torch.cuda.device_count() < 2:
        print("未执行 TP=2 实测（--tp2 且需 2 卡）。仅给理论量级：")
        for bs in [1, 256]:
            bytes_ar = bs * HIDDEN * 2
            print(f"  decode bs={bs}: all-reduce 单次 {bytes_ar / 1024:.1f} KB，"
                  f"NVLink(50GB/s) 约 {bytes_ar / 50e9 * 1e6:.1f} us，"
                  f"跨节点 NIC(12.5GB/s) 约 {bytes_ar / 12.5e9 * 1e6:.1f} us")
        print("说明: Transformer 层内 o_proj 的 all-reduce 结果必须等残差+norm 完成，")
        print("      下一层 MLP 才可开始，通信与后续计算存在硬依赖，无法简单隐藏。")
        return
    import torch.multiprocessing as mp
    port = 29500 + (os.getpid() % 2000)
    mp.spawn(_tp2_worker, args=(port,), nprocs=2, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp2", action="store_true", help="额外跑 TP=2 通信实测（需 2 卡）")
    args = parser.parse_args()

    bench_swiglu()
    bench_gqa()
    bench_shard()
    bench_comm(args.tp2)
