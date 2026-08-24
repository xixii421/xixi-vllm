"""替换 RMSNorm 前后的端到端对比（isolate：同一机器同一流程）。

分别测 eager / decode-CG / prefill-compile 三种模式下的 TTFT 与 decode 吞吐，
确认 flashinfer fused_add_rmsnorm 替换在真实执行路径上无性能倒退。

用法:
    python bench/bench_rmsnorm_e2e.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common import destroy, dummy_ids, make_engine, median, ttft_until_first, warmup, write_result  # noqa: E402
from src import SamplingParams  # noqa: E402


def decode_throughput(engine, bs: int, n: int) -> float:
    """bs 并发 × 32 token，测 tokens/s。"""
    sp = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)
    seqs = [engine.add_request(dummy_ids(16, salt=1000 + i), sp) for i in range(bs)]
    t0 = time.perf_counter()
    while not all(s.is_finished for s in seqs):
        engine.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return bs * 32 / dt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n", type=int, default=8)
    args = p.parse_args()
    torch.cuda.set_device(args.device)

    lines = [
        "# RMSNorm 替换 e2e 对比",
        "",
        "| mode | TTFT@128 (ms) | TTFT@512 (ms) | decode tok/s @bs=16 |",
        "| --- | ---: | ---: | ---: |",
    ]

    modes = [
        ("eager", dict(cg=False, compile=False)),
        ("cg", dict(cg=True, compile=False)),
        ("cg+compile", dict(cg=True, compile=True)),
    ]
    res: dict[str, dict] = {}
    for name, kw in modes:
        engine = make_engine(**kw, prefix="none")
        ttft = {}
        for L in (128, 512):
            warmup(engine, L)
            ttft[L] = median([ttft_until_first(engine, dummy_ids(L, salt=100 + i)) for i in range(args.n)])
        warmup(engine, 16, max_tokens=32)
        tok = median([decode_throughput(engine, 16, 3) for _ in range(3)])
        res[name] = ttft | {"tok": tok}
        destroy(engine)
        print(f"{name}: TTFT128={ttft[128]:.2f}ms TTFT512={ttft[512]:.2f}ms dec={tok:.0f}tok/s", flush=True)

    for name, _ in modes:
        r = res[name]
        row = f"| {name} | {r[128]:.2f} | {r[512]:.2f} | {r['tok']:.0f} |"
        print(row)
        lines.append(row)
    write_result("bench_rmsnorm_e2e.txt", "\n".join(lines))


if __name__ == "__main__":
    main()
