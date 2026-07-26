#!/usr/bin/env python3
"""Standalone prefill benchmark: compare v4 (torch.compile) vs v5 (eager).

Measures raw forward pass time across prompt lengths, bypassing engine
scheduling to focus purely on prefill computation speed.

Usage:
  # v4 with compile
  PYTHONPATH=/home/xingyongwang/code/mycuda/dzyy-vllm/vllm-v4 \
    python bench/bench_prefill_raw.py --version v4 --compile

  # v4 without compile (eager baseline)
  PYTHONPATH=/home/xingyongwang/code/mycuda/dzyy-vllm/vllm-v4 \
    python bench/bench_prefill_raw.py --version v4 --no-compile

  # v5 (always eager for prefill)
  PYTHONPATH=/home/xingyongwang/code/mycuda/dzyy-vllm/vllm-v5 \
    python bench/bench_prefill_raw.py --version v5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

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


def setup_v4(compile_enabled: bool):
    """Load model and optionally wrap with torch.compile."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from src.models.loader import load_model, skip_param_init
    from src.layers.attention import attention_context, BatchDecodeWithPagedKVCacheWrapper
    from src.layers.attention_backend import (
        BackendConfig, FlashInferBackend, compute_paged_kv_metadata,
    )
    from src.utils.context import forward_context

    config = AutoConfig.from_pretrained(TEXT_MODEL, trust_remote_code=True)
    with skip_param_init():
        model = AutoModelForCausalLM.from_pretrained(
            TEXT_MODEL,
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    # Load actual weights
    load_model(model, TEXT_MODEL)
    model = model.to(dtype=torch.float16, device="cuda").eval()

    if compile_enabled:
        print("  Compiling with torch.compile (mode=default)...", file=sys.stderr)
        model = torch.compile(model, dynamic=True, mode="default")

    # Create tiny KV cache for the measurement
    hf = config
    nkv = hf.num_key_value_heads
    hd = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
    layers = hf.num_hidden_layers
    block_size = 256
    num_blocks = 64  # enough for the warmup and measurement
    kv_cache = torch.empty(
        2, layers, num_blocks, block_size, nkv, hd,
        dtype=torch.float16, device="cuda"
    )

    backend = FlashInferBackend(
        BackendConfig(
            num_qo_heads=hf.num_attention_heads,
            num_kv_heads=nkv,
            head_dim=hd,
            block_size=block_size,
            dtype=torch.float16,
            device=torch.device("cuda"),
        )
    )

    return model, kv_cache, backend, forward_context, attention_context, compute_paged_kv_metadata, block_size


def setup_v5():
    """Load model from v5 without compile (always eager for prefill)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from src.models.loader import load_model
    from src.layers.attention import attention_context
    from src.layers.attention_backend import (
        BackendConfig, FlashInferBackend, compute_paged_kv_metadata,
    )
    from src.utils.context import forward_context

    config = AutoConfig.from_pretrained(TEXT_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        TEXT_MODEL,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = model.to(dtype=torch.float16, device="cuda").eval()

    hf = config
    nkv = hf.num_key_value_heads
    hd = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
    layers = hf.num_hidden_layers
    block_size = 256
    num_blocks = 64
    kv_cache = torch.empty(
        2, layers, num_blocks, block_size, nkv, hd,
        dtype=torch.float16, device="cuda"
    )

    backend = FlashInferBackend(
        BackendConfig(
            num_qo_heads=hf.num_attention_heads,
            num_kv_heads=nkv,
            head_dim=hd,
            block_size=block_size,
            dtype=torch.float16,
            device=torch.device("cuda"),
        )
    )

    return model, kv_cache, backend, forward_context, attention_context, compute_paged_kv_metadata, block_size


def run_prefill(model, kv_cache, backend, forward_context, attention_context,
                compute_paged_kv_metadata, block_size, prompt_len: int):
    """Run one prefill forward pass and return wall-clock time."""
    torch.cuda.empty_cache()
    ids = torch.arange(1, prompt_len + 1, dtype=torch.int64, device="cuda")
    positions = torch.arange(prompt_len, dtype=torch.int64, device="cuda")
    n_blocks = (prompt_len + block_size - 1) // block_size
    assert n_blocks <= kv_cache.shape[2], f"n_blocks={n_blocks} > kv={kv_cache.shape[2]}"

    # Build attention metadata
    cu = torch.tensor([0, prompt_len], dtype=torch.int32, device="cuda")
    cu_cpu = torch.tensor([0, prompt_len], dtype=torch.int32, device="cpu")
    sm = torch.arange(prompt_len, dtype=torch.int32, device="cuda")
    bt = torch.full((1, n_blocks), -1, dtype=torch.int32, device="cuda")
    bt[0, :n_blocks] = torch.arange(n_blocks, device="cuda")
    indptr_cpu, idx_gpu, last_cpu = compute_paged_kv_metadata(
        [prompt_len], bt, block_size, "cuda"
    )
    meta = backend.build_metadata(
        1, indptr_cpu, idx_gpu, last_cpu, sm, True, kv_cache, cu_cpu
    )

    sync()
    t0 = time.perf_counter()
    with forward_context(
        is_prefill=True, seqs=[],
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=prompt_len, max_seqlen_k=prompt_len,
        block_tables=bt, slot_mapping=sm,
    ):
        with attention_context(meta):
            _ = model(input_ids=ids, positions=positions)
    sync()
    return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=["v4", "v5"], required=True)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--out", default="")
    args = p.parse_args()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    if args.version == "v4":
        compile_e = not args.no_compile
        label = f"v4_{'compile' if compile_e else 'eager'}"
        t0 = time.perf_counter()
        model, kvc, bk, fc, ac, cpm, bs = setup_v4(compile_enabled=compile_e)
    else:
        label = "v5_eager"
        t0 = time.perf_counter()
        model, kvc, bk, fc, ac, cpm, bs = setup_v5()

    sync()
    load_s = time.perf_counter() - t0
    print(f"  Load: {load_s:.2f}s", file=sys.stderr)
    print(f"  KV cache: {kvc.shape[2]} blocks", file=sys.stderr)

    results = {"version": label, "sweep": {}}

    for plen in SWEEP_LENS:
        print(f"  len={plen:>5d}: ", file=sys.stderr, end="", flush=True)
        times = []

        # Warmup
        for _ in range(WARMUP):
            run_prefill(model, kvc, bk, fc, ac, cpm, bs, plen)

        # Measurement
        for _ in range(ITERS):
            t = run_prefill(model, kvc, bk, fc, ac, cpm, bs, plen)
            times.append(t)

        r = {
            "prompt_len": plen,
            "ms_mean": round(mean(times) * 1000, 2),
            "ms_median": round(statistics.median(times) * 1000, 2),
            "ms_min": round(min(times) * 1000, 2),
            "tok_s": round(plen / mean(times), 0),
        }
        results["sweep"][str(plen)] = r
        print(f"  {r['ms_median']:.2f}ms  {r['tok_s']:.0f} tok/s",
              file=sys.stderr, flush=True)

    # Table
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"  {label}", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    print(f"| {'Prompt':>7s} | {'median(ms)':>12s} | {'mean(ms)':>10s} | {'tok/s':>8s} |", file=sys.stderr)
    print(f"| {'-'*7} | {'-'*12} | {'-'*10} | {'-'*8} |", file=sys.stderr)
    for plen in SWEEP_LENS:
        r = results["sweep"][str(plen)]
        print(f"| {plen:>7d} | {r['ms_median']:>10.2f}ms | {r['ms_mean']:>8.2f}ms | {r['tok_s']:>6.0f} |", file=sys.stderr)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nSaved to {args.out}", file=sys.stderr)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
