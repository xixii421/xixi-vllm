"""单请求 TTFT：eager vs torch.compile prefill。

口径：greedy、max_tokens=1；每个长度先 warmup 3 次再取 n 次中位数。
decode CUDA Graph 两边都开，只隔离 compile。

用法:
    python bench/bench_prefill.py
    python bench/bench_prefill.py --lengths 32,128,512 --n 10
"""
from __future__ import annotations

import argparse

import torch

from common import destroy, dummy_ids, make_engine, median, ttft_until_first, warmup, write_result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lengths", default="32,128,512,1024,2048")
    p.add_argument("--n", type=int, default=10)
    args = p.parse_args()
    torch.cuda.set_device(args.device)
    lengths = [int(x) for x in args.lengths.split(",")]

    lines = [
        "# TTFT  eager vs torch.compile  (greedy, median)",
        "",
        "| prompt | eager (ms) | compile (ms) | speedup |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for s in lines:
        print(s)
    print()

    results: dict[bool, dict[int, float]] = {}
    for compile in (False, True):
        engine = make_engine(cg=True, compile=compile, prefix="none")
        results[compile] = {}
        for L in lengths:
            warmup(engine, L)
            samples = [ttft_until_first(engine, dummy_ids(L, salt=100 + i)) for i in range(args.n)]
            results[compile][L] = median(samples)
        destroy(engine)
        engine = None

    for L in lengths:
        e, c = results[False][L], results[True][L]
        row = f"| {L} | {e:.2f} | {c:.2f} | {e / c:.2f}x |"
        print(row)
        lines.append(row)
    write_result("bench_prefill.txt", "\n".join(lines))


if __name__ == "__main__":
    main()
