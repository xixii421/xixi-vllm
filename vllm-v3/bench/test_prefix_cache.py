#!/usr/bin/env python3
"""prefix cache 命中测试：hash vs radix vs none，KV 数据在不同 token 长度下的效果。

测试方法：
  1. 构造一组「共享前缀 + 不同后缀」的 prompt（前缀命中）。
  2. 依次提交并测量每个请求的 prefill step 时间。
  3. 对比三种 backend 下：第一个请求（冷启动）vs 后续请求（命中）。

注意：block_size=256，前缀命中按整块计。前缀长度取各种奇怪值。

用法：
  python bench/test_prefix_cache.py --backend hash --lens 513 1022 2049 4096
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src import Engine, EngineConfig, SamplingParams


def tok(eng: Engine, text: str) -> list[int]:
    return eng.tokenizer.encode(text, add_special_tokens=False)


def build_shared_prompts(eng: Engine, total: int, n_req: int = 4, hit_frac: float = 0.6) -> list[list[int]]:
    """构造 n_req 个 prompt：共享 hit_frac 比例前缀，其余为随机后缀。"""
    base = tok(eng, "The history of artificial intelligence spans many decades. " * 400)
    common = base[: int(total * hit_frac)]
    out = []
    for i in range(n_req):
        suffix = tok(eng, f"Request number {i}: explain the concept in detail. " * 200)
        suffix = suffix[: total - len(common)]
        # 保证长度对齐
        if len(common) + len(suffix) < total:
            suffix = suffix + [7, 8] * ((total - len(common) - len(suffix) + 1) // 2)
        out.append(common + suffix[: total - len(common)])
    return out


def measure(eng: Engine, prompts: list[list[int]]) -> list[dict]:
    """逐个请求，测每个 prefill 的 wall time + 实际调度 token 数。

    注意：seq.num_cached_tokens 在 deallocate() 时清零，改用 kv.last_num_cached_tokens
    （allocate 时设置，保留到下次 allocate）。
    """
    out = []
    for ids in prompts:
        sp = SamplingParams(max_tokens=1, temperature=0.0)
        seq = eng.add_request(ids, sp)
        t0 = time.perf_counter()
        while not seq.is_finished:
            eng.step()
        dt = (time.perf_counter() - t0) * 1000
        cached = eng.kv.last_num_cached_tokens
        out.append({
            "wall": dt,
            "scheduled": len(ids) - cached,
            "cached": cached,
            "total": len(ids),
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["none", "hash", "radix"], action="append", default=[])
    p.add_argument("--lens", type=int, nargs="+", default=[513, 1022, 2049, 4096])
    p.add_argument("--hit-frac", type=float, default=0.6)
    p.add_argument("--n-req", type=int, default=4)
    p.add_argument("--kv-blocks", type=int, default=128)
    args = p.parse_args()
    backends = args.backend or ["none", "hash", "radix"]

    print(f"{'len':>6s} {'backend':>7s} | {'req0冷':>8s} {'req1':>7s} {'req2':>7s} "
          f"{'req3':>7s} | {'缓存token':>8s} {'省算%':>6s}")
    print("-" * 82)
    for L in args.lens:
        for b in backends:
            eng = Engine(EngineConfig(
                model="/home/xingyongwang/huggingface/Qwen3-0.6B", context_len=16384,
                enforce_eager=True, prefill_cuda_graph=False,
                dtype="auto", prefix_backend=b, max_num_seqs=8,
                num_kvcache_blocks=args.kv_blocks,
                max_num_batched_tokens=8192, gpu_memory_utilization=0.85,
            ))
            prompts = build_shared_prompts(eng, L, args.n_req, args.hit_frac)
            times = measure(eng, prompts)
            saved = times[1]["cached"]
            saved_pct = 100 * saved / L
            row = f"{L:>6d} {b:>7s} | " + " ".join(f"{t['wall']:>7.1f}" for t in times) + \
                  f" | {saved:>7d} {saved_pct:>5.1f}%"
            print(row)
            eng.destroy()


if __name__ == "__main__":
    main()
