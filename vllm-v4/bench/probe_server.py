"""服务器并行体检脚本（可拷到任意多卡机器直接跑）。

用途：回答三个问题
  1) 这台机器几块卡、什么型号、显存多大、卡间能不能 P2P；
  2) NCCL all-reduce 的实际延迟曲线（TP 每层通信的真实成本）；
  3) 以 Qwen 系列 decode 的通信量为基准，估算每个 decode 步的通信总耗时，
     并给出「TP 在本机是否有戏」的判语。

用法：
    python probe_server.py --devices 0,1 --hidden 1024 --layers 28
    # --devices 默认全部卡；--hidden/--layers 用目标模型的隐藏维和层数，
    # 传 --batch 1 即 bs=1 的通信量（也可传 --batch 16 看并发下的量）。

只依赖 torch，不依赖仓库内其他模块，拷到任何有 torch 的机器即可跑。
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _timeit(fn, warmup: int = 10, iters: int = 50) -> float:
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


def single_node_report(devices: list[int]) -> None:
    print("=" * 76)
    print("  1) 硬件清单")
    print("=" * 76)
    for i in devices:
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 2**30
        print(f"  cuda:{i:<2} {name:<32} {cap[0]}.{cap[1]}  {mem:7.1f} GiB")
    n = len(devices)
    if n >= 2:
        print("  卡间 P2P 支持（can_device_access_peer）：")
        for i in devices:
            row = []
            for j in devices:
                if i == j:
                    row.append("  X ")
                else:
                    row.append("OK " if torch.cuda.can_device_access_peer(i, j) else "NS ")
            print("   " + " ".join(f"g{j}" for j in devices))
            print(f"  g{i}  " + "".join(row))
    print("  提示：P2P=OK 表示卡间直连 DMA 可用（配合 nvidia-smi topo -m 看是 NVLink 还是 PCIe）")
    print()


def comm_worker(rank: int, world: int, devices: list[int], sizes_bytes: list[int],
                port: int) -> None:
    torch.cuda.set_device(devices[rank])
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world,
        rank=rank,
    )
    t = torch.device("cuda", devices[rank])
    lats_us: list[float] = []
    if rank == 0:
        print("=" * 76)
        print("  2) NCCL all-reduce 延迟曲线（TP 每层 o_proj/down_proj 各一次 AR）")
        print("=" * 76)
        print(f"  {'消息大小':>10} | {'延迟(us)':>9} | {'有效带宽':>10}")
        print("  " + "-" * 44)
    for nbytes in sizes_bytes:
        x = torch.randn(nbytes // 2, dtype=torch.bfloat16, device=t)
        t_ar = _timeit(lambda: dist.all_reduce(x), warmup=20, iters=100)
        lats_us.append(t_ar * 1000)
        bw = nbytes / (t_ar / 1000) / 2**20  # MB/s，考虑 AR 双向往返
        if rank == 0:
            kb = nbytes / 1024
            print(f"  {kb:>8.0f} KB | {t_ar * 1000:>8.1f} | {bw:>9.1f} MB/s")
    dist.barrier()
    if rank == 0:
        with open(_LAT_FILE, "w") as f:
            for n, u in zip(sizes_bytes, lats_us):
                f.write(f"{n} {u:.2f}\n")
    dist.destroy_process_group()


def read_latencies() -> tuple[list[int], list[float]]:
    sizes, lats = [], []
    with open(_LAT_FILE) as f:
        for line in f:
            n, u = line.split()
            sizes.append(int(n))
            lats.append(float(u))
    return sizes, lats


def estimate_and_verdict(hidden: int, layers: int, batch: int, sizes_bytes: list[int],
                         lats_us: list[float]) -> None:
    per_layer_bytes = 2 * 2 * batch * hidden * 2  # o/down 两次 AR，双向
    step_bytes = per_layer_bytes * layers
    print("=" * 76)
    print("  3) decode 步通信估算与 TP 判语")
    print("=" * 76)
    print(f"  目标模型假设 hidden={hidden}, layers={layers}, batch={batch}")
    print(f"  每层 AR 消息 ≈ {per_layer_bytes / 1024:.0f} KB（o_proj + down_proj 各一次）")
    print(f"  每 decode 步总通信量 ≈ {step_bytes / 1024:.0f} KB")
    # 找到最接近的消息大小的实测延迟
    idx = min(range(len(sizes_bytes)), key=lambda k: abs(sizes_bytes[k] - per_layer_bytes))
    per_ar_us = lats_us[idx]
    step_comm_ms = per_layer_bytes / sizes_bytes[idx] * per_ar_us / 1000 * layers
    print(f"  以 {sizes_bytes[idx] / 1024:.0f}KB 消息的实测延迟 {per_ar_us:.1f}us 折算：")
    print(f"  每步通信串行耗时 ≈ {step_comm_ms:.2f} ms")
    print()
    print("  判语：")
    if step_comm_ms < 1.0:
        print(f"   ✅ {step_comm_ms:.2f}ms 的通信量很小，TP 很有戏（配合 NVLink 或同 NUMA 内 P2P）")
    elif step_comm_ms < 5.0:
        print(f"   ⚠️  {step_comm_ms:.2f}ms 通信可观，TP 只在大模型/强卡（单层计算 > 1ms）下有利")
    else:
        print(f"   ❌ {step_comm_ms:.2f}ms 通信已超过单卡整模型 decode 步耗时，TP 大概率负收益")
    print("   经验法则：只有当「每层单卡计算时间 > 该层 AR 通信时间」时 TP 才划算")
    print()


_LAT_FILE = os.path.join(tempfile.gettempdir(), "probe_server_lats.txt")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--devices", type=str, default=None,
                   help="逗号分隔的 cuda 序号，默认全部卡")
    p.add_argument("--hidden", type=int, default=1024, help="模型隐藏维（Qwen3-0.6B=1024）")
    p.add_argument("--layers", type=int, default=28, help="模型层数")
    p.add_argument("--batch", type=int, default=1, help="decode 并发数")
    args = p.parse_args()

    devices = [int(x) for x in args.devices.split(",")] if args.devices else \
        list(range(torch.cuda.device_count()))
    if not devices:
        print("没有可用 CUDA 设备")
        sys.exit(1)

    print(f"本机: {torch.cuda.get_device_name(devices[0])} 等 {len(devices)} 张卡")
    single_node_report(devices)

    sizes_bytes = [2048, 8192, 32768, 131072, 524288, 2 * 2**20]  # 2K..2M
    if len(devices) >= 2:
        if os.path.exists(_LAT_FILE):
            os.remove(_LAT_FILE)
        port = 29000 + (os.getpid() % 500)
        mp.spawn(comm_worker, args=(len(devices), devices, sizes_bytes, port),
                 nprocs=len(devices), join=True)
        if os.path.exists(_LAT_FILE):
            sizes, lats = read_latencies()
            estimate_and_verdict(args.hidden, args.layers, args.batch, sizes, lats)
        else:
            print("未获取到延迟数据，跳过估值")
    else:
        print("只有 1 张卡，TP 无意义，直接单卡 + CUDA Graph。")
        print("判语：❌ 不要在这台机器上做 TP/PP（没有可并行的卡）")


if __name__ == "__main__":
    main()
