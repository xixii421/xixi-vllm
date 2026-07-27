#!/usr/bin/env python3
"""Prefill TTFT 三路对比: vllm-v3 / ullm / official-vLLM (0.26.0)

Each backend runs at its own best-performance config:
  - vllm-v3: enforce_eager=False (CG on decode, prefill stays eager)
  - ullm:     enforce_eager=False (CG on decode, prefill eager)
  - vllm:     enforce_eager=False (vLLM 0.26 默认开启)
  All: Qwen3-0.6B, prompt_lengths=[128,256,512,1024,2048], gen 1 token

Usage:
  # Run a single backend (from repo root)
  cd /home/xingyongwang/code/mycuda/dzyy-vllm

  # vllm-v3
  PYTHONPATH=vllm-v3 ~/miniconda3/bin/python vllm-v3/bench/bench_prefill_3way.py --backend v3 --out /tmp/v3.json

  # ullm
  ~/miniconda3/bin/python vllm-v3/bench/bench_prefill_3way.py --backend ullm --out /tmp/ullm.json

  # official vLLM
  .venv-vllm/bin/python vllm-v3/bench/bench_prefill_3way.py --backend vllm --out /tmp/vllm.json

  # Or run all three with the shell script
  bash vllm-v3/bench/run_prefill_3way.sh
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
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch

MODEL = "/home/xingyongwang/huggingface/Qwen3-0.6B"
PROMPT_LENS = [128, 256, 512, 1024, 2048]
WARMUP = 3
ITERS = 10

BASE_PROMPT = (
    "Explain continuous batching for LLM inference in clear prose. "
    "Cover scheduling, KV cache paging, and why decode throughput scales "
    "with concurrency. Keep the answer factual and structured. "
)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_prompt(target_tokens: int) -> str:
    full = BASE_PROMPT * 100
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids = tok.encode(full)
    return tok.decode(ids[:target_tokens])


# ══════════════════════════════════════════════════════════════════════
#  vllm-v3 backend
# ══════════════════════════════════════════════════════════════════════

def bench_v3(out_path: str) -> None:
    from src import Engine, EngineConfig

    print("vllm-v3: initializing ...")
    sync()
    t0 = time.perf_counter()
    eng = Engine(EngineConfig(
        model=MODEL,
        context_len=8192,
        enforce_eager=False,        # CG on decode, prefill stays eager
        dtype="auto",
        prefix_backend="none",
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.85,
    ))
    sync()
    init_s = time.perf_counter() - t0
    print(f"  init: {init_s:.1f}s")

    results = {"_meta": {"backend": "vllm-v3", "init_s": init_s}}

    for plen in PROMPT_LENS:
        prompt = make_prompt(plen)
        print(f"\n  prompt={plen}: ", end="", flush=True)
        for _ in range(WARMUP):
            list(eng.generate(prompt, max_new_tokens=1, temperature=0))
        sync()

        m: list[float] = []
        for i in range(ITERS):
            sync()
            t1 = time.perf_counter()
            list(eng.generate(prompt, max_new_tokens=1, temperature=0))
            sync()
            ms = (time.perf_counter() - t1) * 1000
            m.append(ms)
            print(".", end="", flush=True)

        results[f"prefill_{plen}"] = {
            "ttft_mean_ms": round(statistics.mean(m), 3),
            "ttft_median_ms": round(statistics.median(m), 3),
            "ttft_std_ms": round(statistics.stdev(m) if len(m) > 1 else 0, 3),
        }
        print(f" {results[f'prefill_{plen}']['ttft_mean_ms']:.1f}ms")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  ullm backend (async, single request per measure)
# ══════════════════════════════════════════════════════════════════════

def bench_ullm(out_path: str) -> None:
    import asyncio

    async def _run():
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ullm"))
        from ullm import LLM, SamplingParams, EngineConfig

        print("ullm: initializing ...")
        sync()
        t0 = time.perf_counter()
        llm = LLM(EngineConfig(
            model=MODEL,
            gpu_memory_utilization=0.85,
            max_bs=4,
            tp_size=1, pp_size=1,
            enforce_eager=True,
            context_len=8192,
        ))
        sync()
        init_s = time.perf_counter() - t0
        print(f"  init: {init_s:.1f}s")

        results = {"_meta": {"backend": "ullm", "init_s": init_s}}

        for plen in PROMPT_LENS:
            prompt = make_prompt(plen)
            sp = SamplingParams(max_tokens=1, ignore_eos=True, temperature=0)
            print(f"\n  prompt={plen}: ", end="", flush=True)

            # warmup
            for _ in range(WARMUP):
                async for _ in llm.generate(prompt, sp):
                    pass

            m: list[float] = []
            for i in range(ITERS):
                sync()
                t1 = time.perf_counter()
                async for _ in llm.generate(prompt, sp):
                    pass
                sync()
                ms = (time.perf_counter() - t1) * 1000
                m.append(ms)
                print(".", end="", flush=True)

            results[f"prefill_{plen}"] = {
                "ttft_mean_ms": round(statistics.mean(m), 3),
                "ttft_median_ms": round(statistics.median(m), 3),
                "ttft_std_ms": round(statistics.stdev(m) if len(m) > 1 else 0, 3),
            }
            print(f" {results[f'prefill_{plen}']['ttft_mean_ms']:.1f}ms")

        await llm.shutdown()
        return results

    results = asyncio.run(_run())
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  official vLLM (0.26.0)
# ══════════════════════════════════════════════════════════════════════

def bench_vllm(out_path: str) -> None:
    from vllm import LLM, SamplingParams

    print("official vLLM (0.26.0): initializing ...")
    sync()
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL,
        enforce_eager=False,
        max_model_len=8192,
        max_num_seqs=1,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    sync()
    init_s = time.perf_counter() - t0
    print(f"  init: {init_s:.1f}s")

    results = {"_meta": {"backend": "vllm-0.26", "init_s": init_s}}

    for plen in PROMPT_LENS:
        prompt = make_prompt(plen)
        sp = SamplingParams(max_tokens=1, ignore_eos=True, temperature=0)
        print(f"\n  prompt={plen}: ", end="", flush=True)

        # warmup
        for _ in range(WARMUP):
            llm.generate([prompt], sp)

        m: list[float] = []
        for i in range(ITERS):
            sync()
            t1 = time.perf_counter()
            llm.generate([prompt], sp)
            sync()
            ms = (time.perf_counter() - t1) * 1000
            m.append(ms)
            print(".", end="", flush=True)

        results[f"prefill_{plen}"] = {
            "ttft_mean_ms": round(statistics.mean(m), 3),
            "ttft_median_ms": round(statistics.median(m), 3),
            "ttft_std_ms": round(statistics.stdev(m) if len(m) > 1 else 0, 3),
        }
        print(f" {results[f'prefill_{plen}']['ttft_mean_ms']:.1f}ms")

    # cleanup
    del llm
    torch.cuda.empty_cache()

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

BACKENDS = {"v3": bench_v3, "ullm": bench_ullm, "vllm": bench_vllm}

def main() -> None:
    p = argparse.ArgumentParser(description="Prefill TTFT 3-way: v3 / ullm / vLLM")
    p.add_argument("--backend", required=True, choices=list(BACKENDS))
    p.add_argument("--out", required=True, help="Output JSON path")
    args = p.parse_args()

    print(f"=== Prefill TTFT 3-way: {args.backend} ===")
    BACKENDS[args.backend](args.out)


if __name__ == "__main__":
    main()
