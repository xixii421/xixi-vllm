#!/usr/bin/env python3
"""v2 vs v3 公平对比：各种优化条件全开下的 TTFT / TPOT / 吞吐 / e2e。

用法（子进程隔离，PYTHONPATH 决定加载 v2 还是 v3）：
  # v3 全开
  PYTHONPATH=/home/xingyongwang/code/mycuda/dzyy-vllm/vllm-v3 \
    CUDA_VISIBLE_DEVICES=0 ./.venv2/bin/python vllm-v3/bench/bench_v2v3_full.py \
      --version v3 --mode full --out /tmp/v3_full.json
  # v2 全开
  PYTHONPATH=/home/xingyongwang/code/mycuda/dzyy-vllm/vllm-v2 \
    CUDA_VISIBLE_DEVICES=0 ./.venv2/bin/python vllm-v3/bench/bench_v2v3_full.py \
      --version v2 --mode full --out /tmp/v2_full.json

指标口径（与官方对齐）：
  TTFT   = 首 token 时间（wall，含排队，从 step 循环开始计）
  TPOT   = decode 每步时间的 P50（单请求 gen=64 的中位步耗时）
  吞吐   = 总生成 token / e2e wall（并发 bs∈{1,4,8}）
  e2e    = 一批请求从提交到全部完成的 wall
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time

import torch

MODEL = "/home/xingyongwang/huggingface/Qwen3-0.6B"
CONTEXT = 4096


def mean(xs):
    return statistics.mean(xs) if xs else 0.0


def median(xs):
    return statistics.median(xs) if xs else 0.0


def make_prompt(n: int, seed: int) -> list[int]:
    import random
    rng = random.Random(seed)
    return [rng.randint(0, 5000) for _ in range(n)]


def build_engine(version: str, mode: str):
    from src import Engine, EngineConfig, SamplingParams
    if version == "v2":
        # v2 无 CUDA Graph，始终 eager（"full" 与 "eager" 等价）
        cfg = EngineConfig(
            model=MODEL,
            context_len=CONTEXT,
            dtype="auto",
            prefix_backend="hash",
            max_num_seqs=16,
            max_num_batched_tokens=8192,
            mix_prefill_decode=True,
        )
    else:  # v3
        cfg = EngineConfig(
            model=MODEL,
            context_len=CONTEXT,
            enforce_eager=(mode == "eager"),
            dtype="auto",
            prefix_backend="hash",
            num_kvcache_blocks=96,
            mix_prefill_decode=True,
            prefill_cuda_graph=(mode == "full"),
        )
    eng = Engine(cfg)
    return eng, SamplingParams


def measure_ttft(eng, sp_cls, prompt_len: int, iters: int = 5, warmup: int = 2) -> float:
    """单请求 TTFT：gen=1，取中位。warmup 后测稳态。"""
    sp = sp_cls(temperature=0.0, max_tokens=1)
    for _ in range(warmup):
        seq = eng.add_request(make_prompt(prompt_len, seed=900), sp)
        while not seq.is_finished:
            eng.step()
    ttfts = []
    for i in range(iters):
        prompt = make_prompt(prompt_len, seed=100 + i)
        seq = eng.add_request(prompt, sp)
        first = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while not seq.is_finished:
            eng.step()
            if first is None and seq.num_completion_tokens >= 1:
                torch.cuda.synchronize()
                first = time.perf_counter() - t0
        torch.cuda.synchronize()
        ttfts.append(first if first is not None else time.perf_counter() - t0)
    return median(ttfts) * 1000


def measure_tpot_e2e(eng, sp_cls, prompt_len: int = 256, gen: int = 64, iters: int = 5, warmup: int = 1) -> dict:
    """单请求：TPOT(P50) + e2e。decode 阶段逐 step 计时。"""
    sp = sp_cls(temperature=0.0, max_tokens=gen)
    for _ in range(warmup):
        seq = eng.add_request(make_prompt(prompt_len, seed=901), sp)
        while not seq.is_finished:
            eng.step()
    tpots, e2es = [], []
    for i in range(iters):
        prompt = make_prompt(prompt_len, seed=200 + i)
        seq = eng.add_request(prompt, sp)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prev = t0
        while not seq.is_finished:
            eng.step()
            now = time.perf_counter()
            # 只统计 decode 步（num_completion_tokens 已>0 且仍在增长）
            if seq.num_completion_tokens >= 1 and not seq.is_prefill:
                tpots.append(now - prev)
            prev = now
        torch.cuda.synchronize()
        e2es.append(time.perf_counter() - t0)
    return {"tpot_ms": median(tpots) * 1000, "e2e_ms": median(e2es) * 1000}


def measure_concurrent(eng, sp_cls, bs: int, prompt_len: int = 256, gen: int = 128, iters: int = 3, warmup: int = 1) -> dict:
    """并发吞吐 + e2e。"""
    sp = sp_cls(temperature=0.0, max_tokens=gen)
    for _ in range(warmup):
        seqs = [eng.add_request(make_prompt(prompt_len, seed=902), sp) for _ in range(bs)]
        while not all(s.is_finished for s in seqs):
            eng.step()
    walls, toks_list, ttft_sums = [], [], []
    for i in range(iters):
        seqs = [eng.add_request(make_prompt(prompt_len, seed=300 + i * 100 + u), sp)
                for u in range(bs)]
        first_at = [None] * bs
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while not all(s.is_finished for s in seqs):
            eng.step()
            for u, s in enumerate(seqs):
                if first_at[u] is None and s.num_completion_tokens >= 1:
                    first_at[u] = time.perf_counter() - t0
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        toks = sum(s.num_completion_tokens for s in seqs)
        walls.append(wall)
        toks_list.append(toks)
        ttft_sums.append(sum(f for f in first_at) / bs)
    return {
        "tok_s": mean([t / w for t, w in zip(toks_list, walls)]),
        "e2e_ms": median(walls) * 1000,
        "avg_ttft_ms": mean(ttft_sums) * 1000,
        "total_tokens": int(mean(toks_list)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=["v2", "v3"], required=True)
    p.add_argument("--mode", choices=["eager", "full"], required=True)
    p.add_argument("--out", default="")
    args = p.parse_args()

    torch.cuda.empty_cache()
    print(f"GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)
    print(f"== {args.version} {args.mode} ==", file=sys.stderr)

    t0 = time.perf_counter()
    eng, sp_cls = build_engine(args.version, args.mode)
    torch.cuda.synchronize()
    init_s = time.perf_counter() - t0
    print(f"init={init_s:.1f}s", file=sys.stderr)

    res = {"version": args.version, "mode": args.mode, "init_s": round(init_s, 1)}

    # TTFT 扫描（单请求）
    ttft = {}
    for plen in (128, 512, 2048):
        v = measure_ttft(eng, sp_cls, plen)
        ttft[f"len{plen}"] = round(v, 2)
        print(f"  TTFT@len{plen} = {v:.2f}ms", file=sys.stderr)
    res["ttft_ms"] = ttft

    # TPOT + 单请求 e2e
    tpot = measure_tpot_e2e(eng, sp_cls)
    res["tpot_ms"] = round(tpot["tpot_ms"], 2)
    res["single_e2e_ms"] = round(tpot["e2e_ms"], 2)
    print(f"  TPOT = {tpot['tpot_ms']:.2f}ms  single_e2e = {tpot['e2e_ms']:.2f}ms",
          file=sys.stderr)

    # 并发吞吐
    conc = {}
    for bs in (1, 4, 8):
        c = measure_concurrent(eng, sp_cls, bs)
        conc[f"bs{bs}"] = {k: round(v, 2) if isinstance(v, float) else v for k, v in c.items()}
        print(f"  bs{bs}: {c['tok_s']:.1f} tok/s  e2e={c['e2e_ms']:.0f}ms "
              f"avg_ttft={c['avg_ttft_ms']:.1f}ms", file=sys.stderr)
    res["concurrency"] = conc

    if hasattr(eng, "destroy"):
        eng.destroy()
    del eng
    gc.collect()
    torch.cuda.empty_cache()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
    else:
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
