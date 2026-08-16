"""端到端连续批处理吞吐。

默认 16 条变长请求（固定 seed），ignore_eos。
三种 v3 档位隔离 decode CG / compile 的边际收益。

用法:
    python bench/bench_e2e.py
    python bench/bench_e2e.py --mode cg
    python bench/bench_e2e.py --num-seqs 8
"""
from __future__ import annotations

import argparse
import random
import time

import torch

from common import destroy, dummy_ids, drain, make_engine, warmup, write_result

from src import SamplingParams  # noqa: E402

MODES = {
    "eager": dict(cg=False, compile=False),
    "cg": dict(cg=True, compile=False),
    "full": dict(cg=True, compile=True),
}


def workload(n: int, *, min_in=100, max_in=400, min_out=32, max_out=64):
    rng = random.Random(0)
    prompts = [
        dummy_ids(rng.randint(min_in, max_in), salt=i) for i in range(n)
    ]
    max_tokens = [rng.randint(min_out, max_out) for _ in range(n)]
    return prompts, max_tokens


def run_mode(mode: str, prompts, max_tokens) -> tuple[float, int, float]:
    engine = make_engine(
        **MODES[mode],
        prefix="none",
        max_num_seqs=max(16, len(prompts)),
        gpu_mem=0.85,
    )
    warmup(engine, 16, max_tokens=4)
    seqs = [
        engine.add_request(
            p, SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=mt)
        )
        for p, mt in zip(prompts, max_tokens)
    ]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    drain(engine)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    n_out = sum(s.num_completion_tokens for s in seqs)
    destroy(engine)
    return wall, n_out, n_out / wall if wall > 0 else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-seqs", type=int, default=16)
    p.add_argument("--mode", choices=("eager", "cg", "full", "all"), default="all")
    args = p.parse_args()
    torch.cuda.set_device(args.device)

    prompts, max_tokens = workload(args.num_seqs)
    modes = list(MODES) if args.mode == "all" else [args.mode]
    want = sum(max_tokens)

    lines = [
        f"# e2e throughput  (n={args.num_seqs}, target_out≈{want})",
        "",
        "| mode | wall (s) | out tokens | tok/s |",
        "| --- | ---: | ---: | ---: |",
    ]
    for s in lines:
        print(s)
    print()
    for mode in modes:
        wall, n_out, tps = run_mode(mode, prompts, max_tokens)
        row = f"| {mode} | {wall:.2f} | {n_out}/{want} | {tps:.1f} |"
        print(row)
        lines.append(row)
    write_result("bench_e2e.txt", "\n".join(lines))


if __name__ == "__main__":
    main()
