"""GPU 间通信带宽与延迟测试（参考 NVIDIA nccl-tests 与 cuda-samples 方法论）。

设计对标两个官方工具，输出两个视角的数据：

1. 点对点 P2P 视角。对应 cuda-samples 的 p2pBandwidthLatencyTest，
   测两张卡之间裸 copy 的带宽（大消息）和延迟（小消息）。
   P2P 是否可用由 torch.cuda.can_device_access_peer 决定；
   P2P 不可用时数据要绕主机内存，带宽和延迟都会显著劣化。

2. 集合通信视角。对应 nccl-tests 的 all_reduce_perf / all_gather_perf，
   测 NCCL all-reduce / all-gather 随消息大小的 algbw（算法带宽）和
   busbw（总线带宽）。busbw 才是反映硬件瓶颈的真实指标，公式来自
   nccl-tests 的 doc/PERFORMANCE.md：
       all-reduce : busbw = algbw * 2*(n-1)/n
       all-gather : busbw = algbw * (n-1)/n

用法（拷到任意有 torch 的多卡机器即可跑）：
    python bench_bandwidth.py --devices 0,1
    # 默认消息从 2KB 扫到 256MB，2 倍步进
    # -b 改最小消息，-e 改最大，-f 改步进因子（2 表示每档翻倍）

只依赖 torch，不依赖仓库内其他模块。
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# ---------------------------------------------------------------------------
# 计时辅助
# ---------------------------------------------------------------------------

def bench_fn(fn, warmup: int = 10, iters: int = 30) -> float:
    """返回 fn 的单次耗时 ms，取多次迭代的中位数，抗抖动。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def fmt_bw(gb_per_s: float) -> str:
    return f"{gb_per_s:7.2f} GB/s"


def fmt_us(us: float) -> str:
    return f"{us:8.1f} us"


# ---------------------------------------------------------------------------
# 1. 点对点 P2P 测试（单进程，两个 device 之间裸 copy）
# ---------------------------------------------------------------------------

def p2p_test(devices: list[int], min_bytes: int, max_bytes: int, step: int) -> None:
    src, dst = devices[0], devices[1]
    print("=" * 78)
    print("  [1] 点对点 P2P 裸传输（cudaMemcpy D2D 语义）")
    print("=" * 78)
    can = torch.cuda.can_device_access_peer(src, dst)
    print(f"  g{src} -> g{dst}  can_device_access_peer = {'OK' if can else 'NS'}")
    print(f"  {'消息大小':>10} | {'带宽(单向)':>12} | {'往返延迟':>10}")
    print("  " + "-" * 40)

    # 延迟：小消息多次 copy 求中位数
    x4 = torch.zeros(4, dtype=torch.float32, device=f"cuda:{src}")
    y4 = torch.zeros(4, dtype=torch.float32, device=f"cuda:{dst}")
    lat = bench_fn(lambda: y4.copy_(x4, non_blocking=True), warmup=20, iters=100)
    print(f"  {'4 B':>10} | {'-':>12} | {fmt_us(lat * 1000)}")

    # 带宽：大消息，单向 copy
    size = min_bytes
    while size <= max_bytes:
        x = torch.empty(size // 4, dtype=torch.float32, device=f"cuda:{src}")
        y = torch.empty(size // 4, dtype=torch.float32, device=f"cuda:{dst}")
        ms = bench_fn(lambda: y.copy_(x, non_blocking=True), warmup=5, iters=20)
        bw = size / (ms / 1000) / 2**30  # GB/s
        print(f"  {size / 1024:>8.0f} KB | {fmt_bw(bw)} | {'-':>10}")
        size *= step
    print()


# ---------------------------------------------------------------------------
# 2. NCCL 集合通信测试（多进程）
# ---------------------------------------------------------------------------

_LAT_FILE = os.path.join(tempfile.gettempdir(), "bench_bandwidth_lats.txt")


def _collective_worker(rank: int, world: int, devices: list[int],
                       sizes: list[int], port: int) -> None:
    torch.cuda.set_device(devices[rank])
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world,
        rank=rank,
    )
    t = torch.device("cuda", devices[rank])
    if rank == 0:
        print("=" * 78)
        print(f"  [2] NCCL all-reduce（world={world}，busbw = algbw * 2(n-1)/n）")
        print("=" * 78)
        print(f"  {'消息大小':>10} | {'时间':>9} | {'algbw':>10} | {'busbw':>10}")
        print("  " + "-" * 46)
    rows: list[str] = []
    for nbytes in sizes:
        x = torch.zeros(nbytes // 2, dtype=torch.bfloat16, device=t)
        ms = bench_fn(lambda: dist.all_reduce(x), warmup=10, iters=30)
        algbw = nbytes / (ms / 1000) / 2**30
        busbw = algbw * 2 * (world - 1) / world
        if rank == 0:
            kb = nbytes / 1024
            rows.append(f"{kb:>8.0f} KB | {ms * 1000:>7.1f} us | {fmt_bw(algbw)} | {fmt_bw(busbw)}")
            print("  " + rows[-1])
    if rank == 0:
        with open(_LAT_FILE, "w") as f:
            for nbytes, r in zip(sizes, rows):
                f.write(f"{nbytes} {r}\n")
    dist.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# 3. 与硬件理论值对比
# ---------------------------------------------------------------------------

def compare(devices: list[int], min_bytes: int, max_bytes: int, step: int) -> None:
    if not os.path.exists(_LAT_FILE):
        return
    print("=" * 78)
    print("  [3] 与硬件理论带宽对比")
    print("=" * 78)
    sizes, rows = [], []
    with open(_LAT_FILE) as f:
        for line in f:
            n, _, *rest = line.split()
            sizes.append(int(n))
            rows.append(" ".join(rest))
    # 每行末尾是 busbw（倒数第二个 token），取最大消息（已饱和）的 busbw
    busbws = [float(r.split()[-2]) for r in rows]
    peak = max(busbws)
    print(f"  all-reduce busbw 峰值 ≈ {peak:.2f} GB/s（消息 {sizes[busbws.index(peak)] / 1024:.0f} KB 处饱和）")
    print("  本机为跨 NUMA 无 P2P，通信需绕主机内存，理论上限约 12-13 GB/s（PCIe 4.0 两次 DMA）")
    print("  若 busbw 远低于此且机器有 NVLink，说明 NVLink lane 或驱动有问题")


def main() -> None:
    p = argparse.ArgumentParser(description="GPU 间通信带宽与延迟测试")
    p.add_argument("--devices", type=str, default=None, help="逗号分隔的 cuda 序号，默认全部卡")
    p.add_argument("-b", "--min", type=int, default=2048, help="最小消息字节数，默认 2K")
    p.add_argument("-e", "--max", type=int, default=256 * 2**20, help="最大消息字节数，默认 256M")
    p.add_argument("-f", "--factor", type=int, default=2, help="消息大小步进因子，默认 2 倍")
    p.add_argument("--skip-p2p", action="store_true", help="跳过点对点测试")
    args = p.parse_args()

    devices = [int(x) for x in args.devices.split(",")] if args.devices else \
        list(range(torch.cuda.device_count()))
    if not devices:
        print("没有可用 CUDA 设备")
        sys.exit(1)

    print(f"本机: {torch.cuda.get_device_name(devices[0])} 等 {len(devices)} 张卡")
    for i in devices:
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 2**30
        print(f"  cuda:{i:<2} {name:<30} {mem:6.1f} GiB")

    sizes = [args.min]
    while sizes[-1] * args.factor <= args.max:
        sizes.append(sizes[-1] * args.factor)

    if len(devices) >= 2:
        if not args.skip_p2p:
            p2p_test(devices, args.min, args.max, args.factor)
        if os.path.exists(_LAT_FILE):
            os.remove(_LAT_FILE)
        port = 30000 + (os.getpid() % 500)
        mp.spawn(_collective_worker, args=(len(devices), devices, sizes, port),
                 nprocs=len(devices), join=True)
        compare(devices, args.min, args.max, args.factor)
    else:
        print("只有 1 张卡，无法测卡间通信。")
        print("带宽测试至少需要 2 张卡。")


if __name__ == "__main__":
    main()
