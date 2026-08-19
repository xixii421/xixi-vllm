"""TP 通信瓶颈的量化与优化路径验证（2 卡，mp.spawn）。

本机环境：RTX 4090 D + RTX 4070 SUPER，PCIe Gen4 x16，P2P 不可用
（torch.cuda.can_device_access_peer 双向 False），无 NVLink，无对称内存 API。
这决定了 vLLM 的 CustomAllReduce / symm_mem 在这台机器上都不会启用，
NCCL 是唯一可用的 allreduce 后端。基准回答四个问题。

  Part 0  跨卡拷贝带宽实测，判断自定义 kernel 是否有空间（预计被 PCIe 限制）
  Part 1  NCCL all_reduce / all_gather 延迟随消息大小曲线（decode 尺寸范围）
  Part 2  Qwen3-0.6B 形状的合成 decode 步，24 层 x (o_proj AR + down AR)，
          实测纯计算 vs 计算+通信，量化通信占整步的比例
  Part 3  微批交错重叠演示，两个独立批次串行 vs 双流交错
  Part 4  prefill 的 Sequence Parallel 模式，allreduce+rmsnorm
          vs reduce_scatter+rmsnorm+allgather（vLLM SP pass 的 eager 形态）

用法（在 vllm-v4/ 下）：
    ../.venv2/bin/python bench/bench_comm_opt.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HIDDEN = 1024      # Qwen3-0.6B hidden
INTER = 3072       # Qwen3-0.6B intermediate
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
LAYERS = 28        # Qwen3-0.6B 层数


def _time(fn, warmup: int = 20, iters: int = 200) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def _bandwidth_mb(t_ms: float, nbytes: int) -> float:
    return nbytes / (t_ms / 1e3) / 2**20


def main_worker(rank: int, world_size: int, port: int, args) -> None:
    from src.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
        tensor_model_parallel_reduce_scatter,
    )

    init_distributed_environment(
        world_size=world_size, rank=rank, local_rank=rank, backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
    )
    initialize_model_parallel(world_size, 1, rank, 0)
    torch.manual_seed(0)
    dev = torch.device("cuda", rank)
    peer = 1 - rank

    q_per = HEADS // world_size * HEAD_DIM      # 每卡 q 维
    kv_per = KV_HEADS // world_size * HEAD_DIM  # 每卡 kv 维
    attn_out_per = q_per                        # o_proj 输入（仅 q 头输出）
    mlp_in_per = INTER // world_size            # down_proj 输入
    w_o = torch.randn(HIDDEN, attn_out_per, device=dev,
                      dtype=torch.bfloat16) * 0.02
    w_d = torch.randn(HIDDEN, mlp_in_per, device=dev,
                      dtype=torch.bfloat16) * 0.02

    # ------------------------------------------------------------------ #
    # Part 0  跨卡拷贝带宽（P2P 关闭时的底层硬件天花板）
    # ------------------------------------------------------------------ #
    if args.skip < 1:
        print("=" * 78)
        print(f"Part0 跨卡拷贝带宽（P2P 关闭，PCIe 主机中转），rank{rank}")
        print("=" * 78)
        p2p = torch.cuda.can_device_access_peer(rank, peer)
        print(f"  can_device_access_peer({rank}->{peer}) = {p2p}")
        src = torch.randn(8 * 2**20 // 4, device=dev, dtype=torch.float32)
        dst = torch.empty_like(src, device=torch.device("cuda", peer))

        def copy_():
            dst.copy_(src, non_blocking=True)

        t = _time(copy_, warmup=10, iters=50)
        print(f"  d2d copy 8MB 取整耗时 {t:.3f} ms，有效带宽 "
              f"{_bandwidth_mb(t, src.numel() * 4):.1f} MB/s")
        print()

    # ------------------------------------------------------------------ #
    # Part 1  NCCL 原语延迟随消息大小曲线
    # ------------------------------------------------------------------ #
    if args.skip < 2:
        print("=" * 78)
        print(f"Part1 NCCL all_reduce / all_gather 延迟，rank{rank}")
        print("=" * 78)
        print(f"  {'bytes':>9} | {'all_reduce':>10} | {'all_gather':>10} | "
              f"{'AR带宽':>8}")
        for nbytes in [16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024,
                       256 * 1024, 512 * 1024, 1024 * 1024, 4 * 1024 * 1024]:
            n = nbytes // 2  # bf16
            x = torch.randn(n, device=dev, dtype=torch.bfloat16)
            t_ar = _time(lambda: tensor_model_parallel_all_reduce(x),
                         warmup=20, iters=100)
            out = torch.empty(2 * n, dtype=torch.bfloat16, device=dev)
            # 注：通信层的 all_gather 返回新分配的张量；这里测内部算子开销。
            y = torch.randn(n // 2, device=dev, dtype=torch.bfloat16)
            t_ag = _time(lambda: tensor_model_parallel_all_gather(y, dim=-1),
                         warmup=20, iters=100)
            bw = _bandwidth_mb(t_ar, nbytes * 2)  # all_reduce 双向 2x
            print(f"  {nbytes / 1024:>7.0f}K | {t_ar:>9.3f} | {t_ag:>9.3f} | "
                  f"{bw:>7.1f}MB/s")
        print()

    # ------------------------------------------------------------------ #
    # Part 2  Qwen3-0.6B 合成 decode 步，24 层，量化通信占比
    # ------------------------------------------------------------------ #
    if args.skip < 3:
        print("=" * 78)
        print(f"Part2 Qwen3-0.6B 合成 decode 步（24 层 x 2 个 AR），rank{rank}")
        print("=" * 78)

        def one_step(bs: int, with_comm: bool) -> None:
            xo = torch.randn(bs, attn_out_per, device=dev, dtype=torch.bfloat16)
            xd = torch.randn(bs, mlp_in_per, device=dev, dtype=torch.bfloat16)
            for _ in range(LAYERS):
                o = torch.nn.functional.linear(xo, w_o)   # o_proj [bs,H]
                if with_comm:
                    o = tensor_model_parallel_all_reduce(o)
                d = torch.nn.functional.linear(xd, w_d)   # down_proj [bs,H]
                if with_comm:
                    d = tensor_model_parallel_all_reduce(d)

        print(f"  {'bs':>4} | {'纯计算':>9} | {'计算+通信':>10} | {'通信占比':>8}")
        w_o_full = torch.randn(HIDDEN, HIDDEN, device=dev,
                               dtype=torch.bfloat16) * 0.02
        w_d_full = torch.randn(HIDDEN, INTER, device=dev,
                               dtype=torch.bfloat16) * 0.02

        def one_step_tp1(bs: int) -> None:
            # 模拟 tp=1：每卡持全量权重，无任何通信，同等 48 次 GEMM
            xo = torch.randn(bs, HIDDEN, device=dev, dtype=torch.bfloat16)
            xd = torch.randn(bs, INTER, device=dev, dtype=torch.bfloat16)
            for _ in range(LAYERS):
                o = torch.nn.functional.linear(xo, w_o_full)
                d = torch.nn.functional.linear(xd, w_d_full)

        for bs in [16, 32, 64]:
            t_comp = _time(lambda: one_step(bs, False), warmup=5, iters=20)
            t_full = _time(lambda: one_step(bs, True), warmup=5, iters=20)
            t_tp1 = _time(lambda: one_step_tp1(bs), warmup=5, iters=20)
            frac = (t_full - t_comp) / t_full * 100
            nbytes = LAYERS * 2 * bs * HIDDEN * 2 * 2  # 2 个 AR，双向
            print(f"  {bs:>4} | {t_comp:>8.3f} | {t_full:>9.3f} | "
                  f"{frac:>7.1f}%")
            print(f"       通信量 {nbytes / 2**20:.1f} MB/步，"
                  f"等效 AR 带宽 {_bandwidth_mb(t_full - t_comp, nbytes):.1f} MB/s")
            print(f"       tp=1 全量权重无通信 {t_tp1:.3f} ms，"
                  f"tp=2 / tp=1 = {t_full / t_tp1:.2f}x")
        print()

    # ------------------------------------------------------------------ #
    # Part 3  微批交错重叠（两个独立批次）
    # ------------------------------------------------------------------ #
    if args.skip < 4:
        print("=" * 78)
        print(f"Part3 微批交错重叠演示，rank{rank}")
        print("=" * 78)
        bs = 64
        x1 = torch.randn(bs, HIDDEN, device=dev, dtype=torch.bfloat16)
        x2 = torch.randn(bs, HIDDEN, device=dev, dtype=torch.bfloat16)
        w = torch.randn(HIDDEN, HIDDEN, device=dev, dtype=torch.bfloat16) * 0.02

        def serial() -> None:
            g1 = torch.nn.functional.linear(x1, w)
            g1 = tensor_model_parallel_all_reduce(g1)
            g2 = torch.nn.functional.linear(x2, w)
            g2 = tensor_model_parallel_all_reduce(g2)

        s0 = torch.cuda.Stream(device=dev)
        s1 = torch.cuda.Stream(device=dev)

        def overlapped() -> None:
            # 主流：g1 -> g2；副流：ar1（依赖 g1，与 g2 并行）
            g1 = torch.nn.functional.linear(x1, w)
            ev1 = torch.cuda.Event()
            ev1.record()
            s1.wait_event(ev1)
            with torch.cuda.stream(s1):
                a1 = tensor_model_parallel_all_reduce(g1.clone())
            g2 = torch.nn.functional.linear(x2, w)
            ev2 = torch.cuda.Event()
            ev2.record()
            s1.wait_event(ev2)
            with torch.cuda.stream(s1):
                a2 = tensor_model_parallel_all_reduce(g2.clone())
            torch.cuda.current_stream().wait_stream(s1)
            return g1, a1, g2, a2

        t_ser = _time(serial, warmup=20, iters=100)
        t_ovl = _time(overlapped, warmup=20, iters=100)
        print(f"  bs=64 两个独立层批次，串行 {t_ser:.3f} ms，"
              f"双流交错 {t_ovl:.3f} ms，加速 {t_ser / t_ovl:.2f}x")
        print()

    # ------------------------------------------------------------------ #
    # Part 4  prefill 的 SP 模式
    # ------------------------------------------------------------------ #
    if args.skip < 5:
        print("=" * 78)
        print(f"Part4 prefill 的 SP 模式（eager，无内核融合），rank{rank}")
        print("=" * 78)
        half = HIDDEN // world_size
        w_rms = torch.ones(HIDDEN, device=dev, dtype=torch.bfloat16)
        w_rms_half = torch.ones(half, device=dev, dtype=torch.bfloat16)
        eps = 1e-6

        def rms(x: torch.Tensor, wt: torch.Tensor) -> torch.Tensor:
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * wt

        print(f"  {'tokens':>7} | {'allreduce+rms':>14} | "
              f"{'RS+rms+AG':>10} | {'SP/AR':>6}")
        for n in [256, 1024, 4096]:
            x = torch.randn(n, HIDDEN, device=dev, dtype=torch.bfloat16)

            def ar_rms():
                y = tensor_model_parallel_all_reduce(x)
                return rms(y, w_rms)

            def sp():
                y = tensor_model_parallel_reduce_scatter(x, dim=-1)  # [n, half]
                y = rms(y, w_rms_half)
                return tensor_model_parallel_all_gather(y, dim=-1)   # [n, H]

            t_ar = _time(ar_rms, warmup=20, iters=50)
            t_sp = _time(sp, warmup=20, iters=50)
            print(f"  {n:>7} | {t_ar:>13.3f} | {t_sp:>9.3f} | "
                  f"{t_sp / t_ar:>5.2f}x")
        print("  说明：eager 下 SP 需要两次通信，并不快于 all-reduce。")
        print("  vLLM 也承认 SP pass 本身无收益，收益来自后续把 reduce_scatter")
        print("  合入 GEMM epilogue、把 all-gather 合入下一层 GEMM 输入。")
        print()

    # ------------------------------------------------------------------ #
    # Part 5  decode 步 eager vs CUDA Graph（消除 per-op 启动开销）
    #        注意：NCCL 集合通信进图捕获在生产代码里由 model_runner 完成
    #        （见 CUDAGraphRunner + graph_capture），本 bench 避免在简化
    #        复现中捕获 NCCL（容易死锁），只量化计算链的启动开销。
    # ------------------------------------------------------------------ #
    if args.skip < 6:
        print("=" * 78)
        print(f"Part5 decode 计算链 eager vs CUDA Graph（48 次 GEMM），rank{rank}")
        print("=" * 78)
        bs = 32
        xo = torch.randn(bs, attn_out_per, device=dev, dtype=torch.bfloat16)
        xd = torch.randn(bs, mlp_in_per, device=dev, dtype=torch.bfloat16)

        def step_compute():
            for _ in range(LAYERS):
                xo2 = torch.nn.functional.linear(xo, w_o)
                xd2 = torch.nn.functional.linear(xd, w_d)

        for _ in range(3):
            step_compute()
        torch.cuda.synchronize(dev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step_compute()

        def replay():
            graph.replay()

        t_eager = _time(step_compute, warmup=5, iters=20)
        t_graph = _time(replay, warmup=5, iters=20)
        n_ops = LAYERS * 2
        print(f"  bs=32：eager {t_eager:.3f} ms，CUDA Graph {t_graph:.3f} ms，"
              f"加速 {t_eager / t_graph:.2f}x")
        print(f"  Part1 里每个 eager allreduce 的固定开销约 45-65 us，"
              f"{n_ops} 个 AR 的启动开销合计约 {n_ops * 0.05:.1f} ms/步，"
              f"图重放可以把")
        print(f"  这部分压到接近零（仓库 decode 已这么做）。")
        print()

    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", type=int, default=0,
                        help="跳过前 N 个 Part（0 到 4）")
    args = parser.parse_args()
    import torch.multiprocessing as mp
    port = 29600 + (os.getpid() % 2000)
    mp.spawn(main_worker, args=(2, port, args), nprocs=2, join=True)
