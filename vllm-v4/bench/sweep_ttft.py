#!/usr/bin/env python3
"""Sweep prefill TTFT across prompt lengths: v4 (compile) vs v5 (eager).

Both versions use the same measurement method (add_request + step loop).

Usage:
  # v4
  PYTHONPATH=/home/xingyongwang/code/mycuda/dzyy-vllm/vllm-v4 \\
    python bench/sweep_ttft.py --version v4 --out /tmp/ttft_v4.json

  # v5
  PYTHONPATH=/home/xingyongwang/code/mycuda/dzyy-vllm/vllm-v5 \\
    python bench/sweep_ttft.py --version v5 --out /tmp/ttft_v5.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHINDUCTOR_COMPILE_WORKERS", "0")
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")

import torch

TEXT_MODEL = "/home/xingyongwang/huggingface/Qwen3-0.6B"
SWEEP_LENS = [128, 256, 512, 1024, 2048, 4096]
WARMUP = 3
ITERS = 10


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mean(xs):
    return statistics.mean(xs) if xs else 0.0


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else 0.0


def make_prompt_tokens(tokenizer, target_len: int) -> list[int]:
    chunk = ("The quick brown fox jumps over the lazy dog. "
             "Explain continuous batching for LLM inference. ") * 4
    ids: list[int] = []
    while len(ids) < target_len:
        ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
    return ids[:target_len]


# ── Engine factories ──

def make_v4(model_path):
    from src.config import EngineConfig
    from src.engine.engine import Engine
    from src.sampling_params import SamplingParams

    cfg = EngineConfig(
        model=model_path, context_len=8192, enforce_eager=True,
        dtype="float16", max_num_seqs=4, max_num_batched_tokens=8192,
        gpu_memory_utilization=0.85, mix_prefill_decode=True,
        compile_prefill=True, prefix_backend="none",
        num_kvcache_blocks=400,
    )
    eng = Engine(cfg)
    return eng, SamplingParams, "eager_compile"


def make_v5(model_path):
    from src import Engine, EngineConfig, SamplingParams

    cfg = EngineConfig(
        model=model_path, context_len=8192, enforce_eager=True,
        dtype="float16", max_num_seqs=4, max_num_batched_tokens=8192,
        gpu_memory_utilization=0.85, enable_radix=False,
        tp_size=1, pp_size=1,
    )
    eng = Engine(cfg)
    return eng, SamplingParams, "cg+eager"


# ── Unified measurement ──

def measure_ttft(eng, sp_cls, prompt_ids: list[int]) -> float:
    """Return TTFT in seconds for a single prefill, gen=1 token."""
    sp = sp_cls(max_tokens=1, temperature=0.0)
    if hasattr(eng, "add_request") and hasattr(eng, "step"):
        seq = eng.add_request(prompt_ids, sp)
        first = None
        sync()
        t0 = time.perf_counter()
        while not seq.is_finished:
            eng.step()
            if first is None and seq.num_completion_tokens > 0:
                sync()
                first = time.perf_counter() - t0
        sync()
        return first if first is not None else (time.perf_counter() - t0)
    else:
        # fallback for generate() API
        first = None
        sync()
        t0 = time.perf_counter()
        for _ in eng.generate(prompt_ids, max_new_tokens=1, temperature=0.0):
            if first is None:
                sync()
                first = time.perf_counter() - t0
        sync()
        return first if first is not None else (time.perf_counter() - t0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=["v4", "v5"], required=True)
    p.add_argument("--out", default="")
    args = p.parse_args()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print(f"=== {args.version} ===", file=sys.stderr)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"GPU: {gpu}", file=sys.stderr)

    t0 = time.perf_counter()
    if args.version == "v4":
        eng, sp_cls, mode = make_v4(TEXT_MODEL)
    else:
        eng, sp_cls, mode = make_v5(TEXT_MODEL)
    tokenizer = eng.tokenizer
    sync()
    init_s = time.perf_counter() - t0
    print(f"Init: {init_s:.2f}s  mode={mode}", file=sys.stderr)

    results = {"version": args.version, "mode": mode,
               "init_s": round(init_s, 2), "sweep": {}}

    for prompt_len in SWEEP_LENS:
        tokens = make_prompt_tokens(tokenizer, prompt_len)
        actual_len = len(tokens)
        print(f"  len={actual_len:>5d}: ", file=sys.stderr, end="", flush=True)

        ttfts = []
        for i in range(WARMUP + ITERS):
            ttft = measure_ttft(eng, sp_cls, tokens)
            if i >= WARMUP:
                ttfts.append(ttft)

        r = {
            "prompt_len": actual_len,
            "ttft_ms_mean": round(mean(ttfts) * 1000, 2),
            "ttft_ms_median": round(median(ttfts) * 1000, 2),
            "ttft_ms_min": round(min(ttfts) * 1000, 2),
            "prefill_tok_s": round(actual_len / mean(ttfts), 1),
        }
        results["sweep"][str(actual_len)] = r
        print(f"TTFT={r['ttft_ms_median']:.2f}ms  "
              f"tok/s={r['prefill_tok_s']:.0f}",
              file=sys.stderr, flush=True)

    # Table
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"  {args.version} ({mode})", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    h = f"| {'Prompt':>7s} | {'TTFT(med)':>10s} | {'TTFT(mean)':>10s} | {'tok/s':>8s} |"
    print(h, file=sys.stderr)
    print(f"| {'-'*7} | {'-'*10} | {'-'*10} | {'-'*8} |", file=sys.stderr)
    for pl in SWEEP_LENS:
        r = results["sweep"][str(pl)]
        print(f"| {pl:>7d} | {r['ttft_ms_median']:>8.2f}ms | "
              f"{r['ttft_ms_mean']:>8.2f}ms | {r['prefill_tok_s']:>6.0f} |",
              file=sys.stderr)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nSaved to {args.out}", file=sys.stderr)
    else:
        print(json.dumps(results, indent=2))

    eng.destroy()
    del eng
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
