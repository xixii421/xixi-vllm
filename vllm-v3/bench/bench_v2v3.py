#!/usr/bin/env python3
"""Benchmark v2 vs v3: single/multi-concurrency TTFT/Decode/E2E across token sizes.

Usage:
  # v2
  PYTHONPATH=/home/xingyongwang/code/mycuda/dzyy-vllm/vllm-v2 \
    python bench/bench_v2v3.py --version v2 --out /tmp/v2_results.json

  # v3
  PYTHONPATH=/home/xingyongwang/code/mycuda/dzyy-vllm/vllm-v3 \
    python bench/bench_v2v3.py --version v3 --out /tmp/v3_results.json
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

import torch

# Models to test — try Qwen3-0.6B, Qwen3-1.7B if available
MODELS = [
    "/home/xingyongwang/huggingface/Qwen3-0.6B",
    "/home/xingyongwang/huggingface/Qwen3-1.7B",
    "/home/xingyongwang/huggingface/Qwen3-4B",
]

# Single-request token lengths
SINGLE_OUT_LENS = [("TTFT_1tok", 1), ("short_32tok", 32), ("med_128tok", 128), ("long_256tok", 256)]

# Concurrency batch sizes
BATCH_SIZES = [1, 2, 4, 8]

# Warmup / measurement rounds
WARMUP = 4
ITERS = 8


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mean(xs):
    return statistics.mean(xs) if xs else 0.0


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else 0.0


# ── Prompt ──

PROMPT = (
    "Explain continuous batching for LLM inference in clear prose. "
    "Cover scheduling, KV cache paging, and why decode throughput scales with "
    "concurrency. Keep the answer factual and structured. "
) * 8  # ~500 tokens


# ── Engine factories ──

def make_v2(model_path):
    from src import Engine, EngineConfig, SamplingParams

    cfg = EngineConfig(
        model=model_path,
        context_len=2048,
        enforce_eager=True,
        dtype="auto",
        prefix_backend="none",
        max_num_seqs=8,
        max_num_batched_tokens=4096,
    )
    eng = Engine(cfg)
    return eng, SamplingParams, "eager"


def make_v2_compile(model_path):
    from src import Engine, EngineConfig, SamplingParams

    cfg = EngineConfig(
        model=model_path,
        context_len=2048,
        enforce_eager=False,  # enables torch.compile
        dtype="auto",
        prefix_backend="none",
        max_num_seqs=8,
        max_num_batched_tokens=4096,
    )
    eng = Engine(cfg)
    return eng, SamplingParams, "compile"


def make_v3(model_path):
    from src import Engine, EngineConfig, SamplingParams

    cfg = EngineConfig(
        model=model_path,
        context_len=4096,
        enforce_eager=True,
        dtype="auto",
        prefix_backend="none",
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.85,
    )
    eng = Engine(cfg)
    return eng, SamplingParams, "eager"


def make_v3_cudagraph(model_path):
    from src import Engine, EngineConfig, SamplingParams

    cfg = EngineConfig(
        model=model_path,
        context_len=4096,
        enforce_eager=False,  # enables CUDA Graph
        dtype="auto",
        prefix_backend="none",
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.85,
    )
    eng = Engine(cfg)
    return eng, SamplingParams, "cudagraph"


# ── Measurement helpers ──

def measure_single(eng, sp_cls, prompt_text: str, num_new_tokens: int,
                   prompt_len: int) -> dict:
    """Measure TTFT and total decode time for a single request."""
    sp = sp_cls(temperature=0.0, max_tokens=num_new_tokens)
    seq = eng.add_request(prompt_text if isinstance(prompt_text, str) else prompt_text, sp)
    first = None
    sync()
    t0 = time.perf_counter()
    while not seq.is_finished:
        eng.step()
        if first is None and seq.num_completion_tokens > 0:
            sync()
            first = time.perf_counter() - t0
    sync()
    total = time.perf_counter() - t0
    # Verify we got the expected number of tokens
    gen = seq.num_completion_tokens
    return {
        "ttft_s": first if first is not None else total,
        "total_s": total,
        "gen_tokens": gen,
    }


def measure_batch(eng, sp_cls, prompt_text: str, num_new_tokens: int,
                  batch_size: int) -> dict:
    """Measure batched throughput for batch_size concurrent requests."""
    sp = sp_cls(temperature=0.0, max_tokens=num_new_tokens)
    seqs = []
    for _ in range(batch_size):
        seq = eng.add_request(prompt_text, sp)
        seqs.append(seq)

    sync()
    t0 = time.perf_counter()
    pending = set(id(s) for s in seqs)
    while pending:
        eng.step()
        for s in seqs:
            if id(s) in pending and s.is_finished:
                pending.discard(id(s))
    sync()
    total = time.perf_counter() - t0
    total_tokens = sum(s.num_completion_tokens for s in seqs)
    return {
        "total_s": total,
        "total_tokens": total_tokens,
        "batch_size": batch_size,
    }


# ── Main ──

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=["v2_eager", "v2_compile", "v3_eager", "v3_cudagraph"], required=True)
    p.add_argument("--out", default="")
    p.add_argument("--model", default="", help="Model path (defaults to Qwen3-0.6B)")
    args = p.parse_args()

    models_to_test = [args.model] if args.model else MODELS

    for model_path in models_to_test:
        if not os.path.exists(model_path):
            print(f"  SKIP: model not found at {model_path}", file=sys.stderr)
            continue
        model_name = Path(model_path).name
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"  {args.version}  |  {model_name}", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Create engine
        factory_map = {
            "v2": make_v2,
            "v2-compile": make_v2_compile,
            "v3": make_v3,
            "v3-cudagraph": make_v3_cudagraph,
        }
        fn = factory_map[args.version]

        t0 = time.perf_counter()
        eng, sp_cls, mode = fn(model_path)
        sync()
        init_s = time.perf_counter() - t0
        print(f"  Init: {init_s:.2f}s  mode={mode}", file=sys.stderr)

        # Tokenize prompt to get length
        pt = len(eng.tokenizer.encode(PROMPT))
        print(f"  Prompt tokens: {pt}", file=sys.stderr)

        res = {
            "version": args.version,
            "model": model_name,
            "mode": mode,
            "init_s": round(init_s, 2),
            "prompt_tokens": pt,
            "single": {},
            "concurrency": {},
        }

        # ── Single request benchmarks ──
        print(f"\n  --- Single Request ---", file=sys.stderr)
        for lbl, nt in SINGLE_OUT_LENS:
            ttfts, totals = [], []
            for i in range(WARMUP + ITERS):
                d = measure_single(eng, sp_cls, PROMPT, nt, pt)
                if i >= WARMUP:
                    ttfts.append(d["ttft_s"])
                    totals.append(d["total_s"])
            r = {
                "num_tokens": nt,
                "ttft_ms": round(median(ttfts) * 1000, 2),
                "decode_tok_s": round((nt - 1) / (mean(totals) - mean(ttfts)), 2)
                if mean(totals) > mean(ttfts) * 1.02 and nt > 1 else 0,
                "e2e_tok_s": round(nt / mean(totals), 2) if mean(totals) > 0 else 0,
            }
            res["single"][lbl] = r
            print(f"    {lbl:<16s} TTFT={r['ttft_ms']:.1f}ms  decode={r['decode_tok_s']:.1f}t/s  e2e={r['e2e_tok_s']:.1f}t/s",
                  file=sys.stderr, flush=True)

        # ── Concurrency benchmarks ──
        print(f"\n  --- Concurrency ---", file=sys.stderr)
        for bs in BATCH_SIZES:
            rates, walls = [], []
            for i in range(WARMUP + ITERS):
                d = measure_batch(eng, sp_cls, PROMPT, 128, bs)
                if i >= WARMUP:
                    rates.append(d["total_tokens"] / d["total_s"])
                    walls.append(d["total_s"])
            r = {
                "batch_size": bs,
                "tok_s": round(mean(rates), 1),
                "wall_s": round(mean(walls), 3),
            }
            res["concurrency"][f"batch{bs}"] = r
            print(f"    batch{bs}: {r['tok_s']:.1f} tok/s  wall={r['wall_s']:.3f}s",
                  file=sys.stderr, flush=True)

        # Reclaim memory
        if hasattr(eng, "destroy"):
            eng.destroy()
        del eng
        torch.cuda.empty_cache()

        # Output results per model
        if args.out:
            out_path = args.out.replace(".json", f"_{model_name}.json")
            Path(out_path).write_text(json.dumps(res, indent=2))
            print(f"\n  Saved to {out_path}", file=sys.stderr)
        else:
            print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
