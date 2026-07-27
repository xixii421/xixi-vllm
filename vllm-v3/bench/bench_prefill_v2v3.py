#!/usr/bin/env python3
"""Prefill TTFT 专项对比: v2_eager vs v3_eager vs v3_compile_prefill.

在不同 prompt 长度下（128, 256, 512, 1024, 2048）测量 prefill TTFT，
对比：
  - v2  eager（SDPA attention, torch.compile 整模型）
  - v3  eager（FlashInfer paged prefill, Triton kernels, 无 compile）
  - v3  compile_prefill（FlashInfer + torch.compile 整模型, dynamic=True）

CUDA Graph 仅加速 decode，对 prefill 无直接作用，不纳入 prefill 专项对比。

用法:
  # v2
  PYTHONPATH=/path/to/vllm-v2 \
    python vllm-v3/bench/bench_prefill_v2v3.py --version v2 --out /tmp/v2.json

  # v3 eager（不启动 compile_prefill）
  PYTHONPATH=/path/to/vllm-v3 \
    python vllm-v3/bench/bench_prefill_v2v3.py --version v3_eager --out /tmp/v3_eager.json

  # v3 compile_prefill
  PYTHONPATH=/path/to/vllm-v3 \
    python vllm-v3/bench/bench_prefill_v2v3.py --version v3 --out /tmp/v3_compile.json
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

MODEL = "/home/xingyongwang/huggingface/Qwen3-0.6B"

# Prompt lengths to test (tokens)
PROMPT_LENS = [128, 256, 512, 1024, 2048]

# Benchmark rounds
WARMUP = 3
ITERS = 10

# Base prompt to repeat
BASE_PROMPT = (
    "Explain continuous batching for LLM inference in clear prose. "
    "Cover scheduling, KV cache paging, and why decode throughput scales "
    "with concurrency. Keep the answer factual and structured. "
)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mean(xs: list[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def truncated_prompt(target_tokens: int, tokenizer) -> str:
    """Generate a prompt of approximately target_tokens tokens."""
    full = BASE_PROMPT * 100
    ids = tokenizer.encode(full)
    return tokenizer.decode(ids[:target_tokens])


def measure_prefill(engine, prompt: str, max_new: int = 1) -> float:
    """Measure prefill TTFT by generating just 1 token (returns TTFT in ms)."""
    if hasattr(engine, "_decache_prompt"):
        engine._decache_prompt()

    torch.cuda.empty_cache()
    sync()
    t0 = time.perf_counter()
    # Generate exactly 1 token
    it = engine.generate(prompt, max_new_tokens=max_new, temperature=0.0)
    try:
        next(it)
        sync()
        ttft = time.perf_counter() - t0
    except StopIteration:
        sync()
        ttft = time.perf_counter() - t0
    return ttft * 1000.0  # ms


# ── Engine factories ──────────────────────────────────────────────────────────


def get_tokenizer_v2(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def get_tokenizer_v3(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def make_v2(model_path: str):
    from src import Engine, EngineConfig

    return Engine(
        EngineConfig(
            model=model_path,
            context_len=4096,
            enforce_eager=True,
            dtype="auto",
            prefix_backend="none",
            max_num_seqs=8,
            max_num_batched_tokens=4096,
            block_size=16,
        )
    )


def make_v3_eager(model_path: str):
    from src import Engine, EngineConfig

    return Engine(
        EngineConfig(
            model=model_path,
            context_len=4096,
            enforce_eager=True,
            compile_prefill=False,
            dtype="auto",
            prefix_backend="none",
            max_num_seqs=8,
            max_num_batched_tokens=8192,
            kvcache_block_size=256,
        )
    )


def make_v3_compile(model_path: str):
    from src import Engine, EngineConfig

    return Engine(
        EngineConfig(
            model=model_path,
            context_len=4096,
            enforce_eager=True,
            compile_prefill=True,
            dtype="auto",
            prefix_backend="none",
            max_num_seqs=8,
            max_num_batched_tokens=8192,
            kvcache_block_size=256,
        )
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="Prefill TTFT benchmark: v2 vs v3")
    p.add_argument("--version", required=True, choices=("v2", "v3_eager", "v3"))
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--prompt-lens", nargs="+", type=int, default=PROMPT_LENS)
    p.add_argument("--model", default=MODEL)
    args = p.parse_args()

    model_path = args.model
    label = args.version
    prompt_lens = args.prompt_lens

    print(f"Benchmark: {label}  |  model={Path(model_path).name}")
    print(f"Warmup={WARMUP}  Iters={ITERS}  PromptLens={prompt_lens}")
    print("-" * 60)

    # Build engine
    print(f"Initializing {label} engine ...")
    sync()
    t0 = time.perf_counter()
    if label == "v2":
        engine = make_v2(model_path)
        tok = get_tokenizer_v2(model_path)
    elif label == "v3_eager":
        engine = make_v3_eager(model_path)
        tok = get_tokenizer_v3(model_path)
    else:  # v3 (compile_prefill)
        engine = make_v3_compile(model_path)
        tok = get_tokenizer_v3(model_path)
    sync()
    init_time = time.perf_counter() - t0
    print(f"Engine ready ({init_time:.1f}s)")
    print("-" * 60)

    results: dict[str, dict] = {}
    results["_meta"] = {
        "version": label,
        "model": Path(model_path).name,
        "init_s": init_time,
        "warmup": WARMUP,
        "iters": ITERS,
    }

    for plen in prompt_lens:
        prompt = truncated_prompt(plen, tok)
        actual_len = len(tok.encode(prompt))
        print(f"\nPrompt: target={plen}  actual={actual_len}")

        # Warmup
        for i in range(WARMUP):
            ttft = measure_prefill(engine, prompt)
            print(f"  warmup {i+1}/{WARMUP}: {ttft:.2f} ms")

        # Measurement
        measurements: list[float] = []
        for i in range(ITERS):
            ttft = measure_prefill(engine, prompt)
            measurements.append(ttft)
            print(f"  iter {i+1}/{ITERS}: {ttft:.2f} ms")

        avg = mean(measurements)
        med = statistics.median(measurements)
        std = statistics.stdev(measurements) if len(measurements) > 1 else 0.0

        results[f"prefill_{plen}"] = {
            "prompt_tokens": actual_len,
            "ttft_mean_ms": round(avg, 3),
            "ttft_median_ms": round(med, 3),
            "ttft_std_ms": round(std, 3),
            "ttft_min_ms": round(min(measurements), 3),
            "ttft_max_ms": round(max(measurements), 3),
            "raw_measurements": [round(v, 3) for v in measurements],
        }
        print(f"  => mean={avg:.2f} ms  median={med:.2f} ms  std={std:.2f} ms")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Prompt':>8}  {'TTFT Mean':>10}  {'TTFT Median':>10}  {'Min':>8}")
    print("-" * 70)
    for plen in prompt_lens:
        r = results[f"prefill_{plen}"]
        print(
            f"{plen:>8}  {r['ttft_mean_ms']:>10.2f}  {r['ttft_median_ms']:>10.2f}  {r['ttft_min_ms']:>8.2f}"
        )


if __name__ == "__main__":
    main()
