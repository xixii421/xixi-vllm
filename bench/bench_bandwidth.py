"""GPU 间通信带宽与延迟（对标 nccl-tests / cuda-samples）。

两个视角：
1. 点对点 P2P：裸 D2D copy 的延迟（小消息）与带宽（大消息）
2. NCCL all-reduce：algbw / busbw（busbw = algbw * 2(n-1)/n）

用法（在仓库根目录）：
    python bench/bench_bandwidth.py --devices 0,1
    python bench/bench_bandwidth.py --pairs PIX:0,1;PXB:2,3;SYS:0,2
    python bench/bench_bandwidth.py --pairs PIX:0,1 --skip-nccl

只依赖 torch。P2P 使用主机侧计时 + 双端 synchronize，避免跨 device
CUDA Event 计时偏差。
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def bench_event(fn, warmup: int = 10, iters: int = 30) -> float:
    """单 device 上的 CUDA Event 计时，返回 ms 中位数。"""
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


def bench_host(fn, devices: list[int], warmup: int = 10, iters: int = 30) -> float:
    """跨 device copy：主机计时 + 两端 synchronize，返回 ms 中位数。"""
    for d in devices:
        torch.cuda.synchronize(d)
    for _ in range(warmup):
        fn()
        for d in devices:
            torch.cuda.synchronize(d)
    times: list[float] = []
    for _ in range(iters):
        for d in devices:
            torch.cuda.synchronize(d)
        t0 = time.perf_counter()
        fn()
        for d in devices:
            torch.cuda.synchronize(d)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def fmt_bw(gb_per_s: float) -> str:
    return f"{gb_per_s:7.2f} GB/s"


def fmt_us(us: float) -> str:
    return f"{us:8.1f} us"


def p2p_test(
    devices: list[int],
    min_bytes: int,
    max_bytes: int,
    step: int,
    label: str = "",
) -> None:
    src, dst = devices[0], devices[1]
    tag = f" ({label})" if label else ""
    print("=" * 78)
    print(f"  [1] 点对点 P2P 裸传输{tag}  g{src} -> g{dst}")
    print("=" * 78)
    can = torch.cuda.can_device_access_peer(src, dst)
    print(f"  can_device_access_peer = {'OK' if can else 'NS'}")
    print(f"  {'消息大小':>10} | {'带宽(单向)':>12} | {'延迟':>10}")
    print("  " + "-" * 40)

    sync_devs = [src, dst]
    x4 = torch.zeros(4, dtype=torch.float32, device=f"cuda:{src}")
    y4 = torch.zeros(4, dtype=torch.float32, device=f"cuda:{dst}")
    lat_ms = bench_host(
        lambda: y4.copy_(x4, non_blocking=True),
        sync_devs,
        warmup=20,
        iters=100,
    )
    print(f"  {'4 B':>10} | {'-':>12} | {fmt_us(lat_ms * 1000)}")

    size = min_bytes
    while size <= max_bytes:
        x = torch.empty(size // 4, dtype=torch.float32, device=f"cuda:{src}")
        y = torch.empty(size // 4, dtype=torch.float32, device=f"cuda:{dst}")
        ms = bench_host(
            lambda: y.copy_(x, non_blocking=True),
            sync_devs,
            warmup=5,
            iters=20,
        )
        bw = size / (ms / 1000) / 2**30
        print(f"  {size / 1024:>8.0f} KB | {fmt_bw(bw)} | {'-':>10}")
        del x, y
        torch.cuda.empty_cache()
        size *= step
    print()


_LAT_FILE = os.path.join(tempfile.gettempdir(), "bench_bandwidth_lats.txt")


def _collective_worker(
    rank: int, world: int, devices: list[int], sizes: list[int], port: int
) -> None:
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
        x = torch.zeros(max(nbytes // 2, 1), dtype=torch.bfloat16, device=t)
        ms = bench_event(lambda: dist.all_reduce(x), warmup=10, iters=30)
        algbw = nbytes / (ms / 1000) / 2**30
        busbw = algbw * 2 * (world - 1) / world
        if rank == 0:
            kb = nbytes / 1024
            rows.append(
                f"{kb:>8.0f} KB | {ms * 1000:>7.1f} us | {fmt_bw(algbw)} | {fmt_bw(busbw)}"
            )
            print("  " + rows[-1])
        del x
        torch.cuda.empty_cache()
    if rank == 0:
        with open(_LAT_FILE, "w") as f:
            for nbytes, r in zip(sizes, rows):
                f.write(f"{nbytes} {r}\n")
    dist.barrier()
    dist.destroy_process_group()


def compare() -> None:
    if not os.path.exists(_LAT_FILE):
        return
    print("=" * 78)
    print("  [3] all-reduce busbw 摘要")
    print("=" * 78)
    sizes, busbws = [], []
    with open(_LAT_FILE) as f:
        for line in f:
            parts = line.split()
            sizes.append(int(parts[0]))
            busbws.append(float(parts[-2]))
    peak = max(busbws)
    print(
        f"  busbw 峰值 ≈ {peak:.2f} GB/s"
        f"（消息 {sizes[busbws.index(peak)] / 1024:.0f} KB）"
    )
    print("  解读：对比硬件标称用 busbw；PCIe 机常见饱和约数十 GB/s，NVLink 为数百 GB/s")
    print()


def parse_pairs(text: str) -> list[tuple[str, list[int]]]:
    """PIX:0,1;PXB:2,3 -> [('PIX',[0,1]), ...]"""
    out: list[tuple[str, list[int]]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"bad --pairs item {chunk!r}, expect LABEL:a,b")
        label, devs = chunk.split(":", 1)
        devices = [int(x) for x in devs.split(",")]
        if len(devices) < 2:
            raise ValueError(f"pair {label} needs >=2 devices")
        out.append((label.strip(), devices))
    return out


def run_nccl(devices: list[int], sizes: list[int]) -> None:
    if os.path.exists(_LAT_FILE):
        os.remove(_LAT_FILE)
    port = 30000 + (os.getpid() % 500) + sum(devices)
    mp.spawn(
        _collective_worker,
        args=(len(devices), devices, sizes, port),
        nprocs=len(devices),
        join=True,
    )
    compare()


def main() -> None:
    p = argparse.ArgumentParser(description="GPU 间通信带宽与延迟测试")
    p.add_argument("--devices", type=str, default=None, help="单对：逗号分隔 cuda 序号")
    p.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="多对：LABEL:a,b;LABEL:c,d  例 PIX:0,1;PXB:2,3;SYS:0,2",
    )
    p.add_argument("-b", "--min", type=int, default=2048, help="最小消息字节，默认 2K")
    p.add_argument("-e", "--max", type=int, default=64 * 2**20, help="最大消息字节，默认 64M")
    p.add_argument("-f", "--factor", type=int, default=2, help="消息步进因子，默认 2")
    p.add_argument("--skip-p2p", action="store_true")
    p.add_argument("--skip-nccl", action="store_true")
    args = p.parse_args()

    if args.pairs:
        jobs = parse_pairs(args.pairs)
    else:
        devices = (
            [int(x) for x in args.devices.split(",")]
            if args.devices
            else list(range(torch.cuda.device_count()))
        )
        jobs = [("", devices)]

    if not jobs or len(jobs[0][1]) < 2:
        print("需要至少 2 张卡（--devices 或 --pairs）")
        sys.exit(1)

    all_devs = sorted({d for _, ds in jobs for d in ds})
    print(f"本机: {torch.cuda.get_device_name(all_devs[0])} 等可见卡")
    for i in all_devs:
        mem = torch.cuda.get_device_properties(i).total_memory / 2**30
        print(f"  cuda:{i:<2} {torch.cuda.get_device_name(i):<30} {mem:6.1f} GiB")
    print()

    sizes = [args.min]
    while sizes[-1] * args.factor <= args.max:
        sizes.append(sizes[-1] * args.factor)

    for label, devices in jobs:
        pair = devices[:2]
        if not args.skip_p2p:
            p2p_test(pair, args.min, args.max, args.factor, label=label)
        if not args.skip_nccl:
            if label:
                print(f"  --- NCCL on {label} {pair} ---")
            run_nccl(pair if len(devices) == 2 else devices, sizes)


if __name__ == "__main__":
    main()
