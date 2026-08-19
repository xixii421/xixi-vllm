"""TP 正负收益的交叉点扫描（2 卡，mp.spawn）。

回答一个问题，在当前硬件（4090D + 4070SUPER，无 NVLink/P2P）下，
TP 在什么模型规模 × 并发下才从负收益翻正。

用真实 Qwen3 系列的 (hidden, heads, kv, inter, layers) 参数，模拟每层
decode 的 qkv/o/down 三个 GEMM 加 o/down 两个 all-reduce：

    tp1: 全量权重，无通信
    tp2: 权重按输出/输入维对半切，o_proj 与 down_proj 输出各做一次 all-reduce

模型参数表（真实配置，head_dim=128）：

    0.6B  h=1024  heads=16  kv=8   inter=3072   layers=28
    1.7B  h=2048  heads=24  kv=4   inter=6272   layers=28
    4B    h=2560  heads=32  kv=8   inter=7680   layers=36
    8B    h=3584  heads=28  kv=4   inter=18944  layers=36
    14B   h=5120  heads=40  kv=8   inter=27648  layers=40

用法（在 vllm-v4/ 下）：
    ../.venv2/bin/python bench/bench_tp_scale.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HEAD_DIM = 128

MODELS = [
    ("0.6B", 1024, 16, 8, 3072, 28),
    ("1.7B", 2048, 24, 4, 6272, 28),
    ("4B", 2560, 32, 8, 7680, 36),
    ("8B", 3584, 28, 4, 18944, 36),
    ("14B", 5120, 40, 8, 27648, 40),
]
BS_LIST = [1, 16, 64, 128]


def _time(fn, warmup: int = 5, iters: int = 20) -> float:
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


def main_worker(rank: int, world_size: int, port: int) -> None:
    from src.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_reduce,
    )

    init_distributed_environment(
        world_size=world_size, rank=rank, local_rank=rank, backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
    )
    initialize_model_parallel(world_size, 1, rank, 0)
    torch.manual_seed(0)
    dev = torch.device("cuda", rank)

    print("=" * 86)
    print(f"TP 正负收益交叉点扫描（rank{rank}，{torch.cuda.get_device_name(rank)}）")
    print("=" * 86)
    print(f"  {'模型':>6} | {'bs':>4} | {'tp1单步':>8} | {'tp2单步':>8} | "
          f"{'tp2/tp1':>8} | {'AR量/层':>8}")
    print("-" * 86)

    for name, hidden, heads, kv, inter, layers in MODELS:
        qdim = heads * HEAD_DIM            # tp1 o_proj 输入维（q 输出维）
        kv_dim = kv * HEAD_DIM
        qkv_out = qdim + 2 * kv_dim        # tp1 qkv 输出维

        # tp1 全量权重
        w_qkv1 = torch.randn(qkv_out, hidden, device=dev, dtype=torch.bfloat16) * 0.02
        w_o1 = torch.randn(hidden, qdim, device=dev, dtype=torch.bfloat16) * 0.02
        w_d1 = torch.randn(hidden, inter, device=dev, dtype=torch.bfloat16) * 0.02
        # tp2 分片权重
        w_qkv2 = torch.randn(qkv_out // 2, hidden, device=dev, dtype=torch.bfloat16) * 0.02
        w_o2 = torch.randn(hidden, qdim // 2, device=dev, dtype=torch.bfloat16) * 0.02
        w_d2 = torch.randn(hidden, inter // 2, device=dev, dtype=torch.bfloat16) * 0.02

        def tp1(bs: int) -> None:
            x = torch.randn(bs, hidden, device=dev, dtype=torch.bfloat16)
            xo = torch.randn(bs, qdim, device=dev, dtype=torch.bfloat16)
            xd = torch.randn(bs, inter, device=dev, dtype=torch.bfloat16)
            for _ in range(layers):
                torch.nn.functional.linear(x, w_qkv1)   # qkv 输出不通信
                torch.nn.functional.linear(xo, w_o1)
                torch.nn.functional.linear(xd, w_d1)

        def tp2(bs: int) -> None:
            x = torch.randn(bs, hidden, device=dev, dtype=torch.bfloat16)
            xo = torch.randn(bs, qdim // 2, device=dev, dtype=torch.bfloat16)
            xd = torch.randn(bs, inter // 2, device=dev, dtype=torch.bfloat16)
            for _ in range(layers):
                torch.nn.functional.linear(x, w_qkv2)
                tensor_model_parallel_all_reduce(
                    torch.nn.functional.linear(xo, w_o2))
                tensor_model_parallel_all_reduce(
                    torch.nn.functional.linear(xd, w_d2))

        for bs in BS_LIST:
            t1 = _time(lambda: tp1(bs), warmup=3, iters=15)
            t2 = _time(lambda: tp2(bs), warmup=3, iters=15)
            ar_bytes = 2 * bs * hidden * 2  # 每层 2 次 all-reduce，双向
            print(f"  {name:>6} | {bs:>4} | {t1:>7.3f} | {t2:>7.3f} | "
                  f"{t2 / t1:>7.2f}x | {ar_bytes / 1024:>6.0f}KB")
        print()

    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    port = 29700 + (os.getpid() % 2000)
    mp.spawn(main_worker, args=(2, port), nprocs=2, join=True)
