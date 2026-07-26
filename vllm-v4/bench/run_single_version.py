#!/usr/bin/env python3
"""Single-version benchmark: called with --version v4|v5.

Driver sets PYTHONPATH to the correct repo root.
Outputs JSON result to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

# Prevent torch.compile workers from reserving GPU memory (they compete with KV cache)
os.environ.setdefault("TORCHINDUCTOR_COMPILE_WORKERS", "0")
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")

import torch

TEXT_MODEL = "/home/xingyongwang/huggingface/Qwen3-0.6B"
VL_MODEL   = "/home/xingyongwang/huggingface/Qwen3-VL-2B-Instruct"
PROMPT     = ("Explain continuous batching for LLM inference in clear prose. "
              "Cover scheduling, KV cache paging, and why decode throughput scales with "
              "concurrency. Keep the answer factual and structured. ") * 8


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mean(xs):
    return statistics.mean(xs) if xs else 0.0


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else 0.0


def peak_gb():
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def pack(ttft, total, gen_n, prompt_n, note=""):
    if total > ttft * 1.02 and gen_n > 1:
        ds = total - ttft
        dt = (gen_n - 1) / ds
    else:
        ds = total
        dt = gen_n / total if total > 0 else 0.0
    return {"ttft_s": round(ttft, 4), "total_s": round(total, 4),
            "decode_tok_s": round(dt, 2),
            "e2e_tok_s": round(gen_n / total, 2) if total > 0 else 0,
            "gen_tokens": gen_n, "prompt_tokens": prompt_n, "note": note}


# =======================================================================
#  v4 implementation
# =======================================================================

def make_v4(model_path, max_ns=16, gpu_util=0.85, ctx=4096, num_blocks_override=0):
    from src.config import EngineConfig
    from src.engine.engine import Engine
    from src.sampling_params import SamplingParams

    cfg = EngineConfig(model=model_path, context_len=ctx, enforce_eager=False,
                       dtype="float16", max_num_seqs=max_ns,
                       max_num_batched_tokens=8192, gpu_memory_utilization=gpu_util,
                       mix_prefill_decode=True, compile_prefill=True, prefix_backend="none",
                       num_kvcache_blocks=num_blocks_override)

    # Debug: compute what auto_num_blocks would give
    free_before, total_before = torch.cuda.mem_get_info()
    print(f"  [v4 debug] total={total_before/1e9:.1f}GB free={free_before/1e9:.1f}GB util={gpu_util}", file=sys.stderr)

    eng = Engine(cfg)

    # Check actual num_blocks allocated
    nblocks = eng.config.num_kvcache_blocks
    print(f"  [v4 debug] allocated KV blocks={nblocks}", file=sys.stderr)
    return eng, SamplingParams, {"cg": eng.model_runner.cuda_graph_runner is not None,
                                  "compile": eng.model_runner._prefill_model is not None}


def bench_v4_single(eng, sp_cls, prompt, max_new, ntoks):
    sp = sp_cls(max_tokens=max_new, temperature=0.0)
    seq = eng.add_request(prompt, sp)
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
    return pack(first or total, total, seq.num_completion_tokens, ntoks)


def bench_v4_batch(eng, sp_cls, prompt, max_new, ntoks, bs):
    sp = sp_cls(max_tokens=max_new, temperature=0.0)
    seqs = [eng.add_request(prompt, sp) for _ in range(bs)]
    sync()
    t0 = time.perf_counter()
    while not all(s.is_finished for s in seqs):
        eng.step()
    sync()
    total = time.perf_counter() - t0
    gens = sum(s.num_completion_tokens for s in seqs)
    return pack(total, total, gens, ntoks, f"batch{bs}")


# =======================================================================
#  v5 implementation
# =======================================================================

def make_v5(model_path, max_ns=16, gpu_util=0.85, ctx=4096, num_blocks_override=0):
    from src import Engine, EngineConfig, SamplingParams

    cfg = EngineConfig(model=model_path, context_len=ctx, enforce_eager=False,
                       dtype="float16", max_num_seqs=max_ns,
                       max_num_batched_tokens=8192, gpu_memory_utilization=gpu_util,
                       enable_radix=False, tp_size=1, pp_size=1)
    eng = Engine(cfg)
    return eng, SamplingParams, {"cg": True, "compile": False}


def bench_v5_single(eng, sp_cls, prompt, max_new, ntoks):
    first = None
    n = 0
    sync()
    t0 = time.perf_counter()
    for _ in eng.generate(prompt, max_new_tokens=max_new, temperature=0.0):
        n += 1
        if first is None:
            sync()
            first = time.perf_counter() - t0
    sync()
    total = time.perf_counter() - t0
    return pack(first or total, total, n, ntoks)


def bench_v5_batch(eng, sp_cls, prompt, max_new, ntoks, bs):
    sp = sp_cls(max_tokens=max_new, temperature=0.0)
    sync()
    t0 = time.perf_counter()
    out = eng.generate_batch([prompt] * bs, sp)
    sync()
    total = time.perf_counter() - t0
    gens = sum(len(eng.tokenizer.encode(t)) for t in out)
    return pack(total, total, gens, ntoks, f"batch{bs}")


# =======================================================================
#  Benchmark runner
# =======================================================================

DO_SINGLE = [("TTFT_1tok", 1), ("short_32tok", 32), ("med_128tok", 128), ("long_256tok", 256)]
BATCH_SIZES = [1, 2, 4, 8]


def run_text(make_fn, single_fn, batch_fn):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    # Use explicit num_kvcache_blocks for v4 to ensure fair comparison with v5
    # (both should have ~580 blocks at gpu_util=0.85)
    eng, sp_cls, info = make_fn(TEXT_MODEL, num_blocks_override=0)
    sync()
    load_s = time.perf_counter() - t0
    print(f"  init={load_s:.2f}s  cg={info['cg']}  compile={info['compile']}", file=sys.stderr, flush=True)

    res = {"info": {"engine_init_s": round(load_s, 2),
                    "peak_mem_init_gb": round(peak_gb(), 2),
                    "cg": info["cg"], "compile": info["compile"]},
           "single": {}, "concurrency": {}}

    pt = len(eng.tokenizer.encode(PROMPT))

    for lbl, nt in DO_SINGLE:
        ttfts, totals = [], []
        for i in range(12):
            d = single_fn(eng, sp_cls, PROMPT, nt, pt)
            if i >= 4:
                ttfts.append(d["ttft_s"])
                totals.append(d["total_s"])
        r = {"ttft_ms": round(median(ttfts) * 1000, 2),
             "decode_tok_s": round((nt - 1) / (mean(totals) - mean(ttfts)), 2)
             if mean(totals) > mean(ttfts) * 1.02 and nt > 1 else 0,
             "e2e_tok_s": round(nt / mean(totals), 2) if mean(totals) > 0 else 0}
        res["single"][lbl] = r
        print(f"  {lbl:<16s} TTFT={r['ttft_ms']:.1f}ms  decode={r['decode_tok_s']:.1f}t/s",
              file=sys.stderr, flush=True)

    for bs in BATCH_SIZES:
        rates, walls = [], []
        for i in range(12):
            d = batch_fn(eng, sp_cls, PROMPT, 128, pt, bs)
            if i >= 4:
                rates.append(d["e2e_tok_s"])
                walls.append(d["total_s"])
        r = {"e2e_tok_s": round(mean(rates), 2),
             "e2e_latency_s": round(mean(walls), 4),
             "peak_mem_gb": round(peak_gb(), 2)}
        res["concurrency"][f"bs{bs}"] = r
        print(f"  bs={bs:<3d}  {r['e2e_tok_s']:.1f}t/s  {r['e2e_latency_s']*1000:.0f}ms",
              file=sys.stderr, flush=True)

    eng.destroy()
    del eng
    torch.cuda.empty_cache()
    return res


def run_mm(make_fn, single_fn, _batch_fn):
    from PIL import Image
    import numpy as np
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    Y, X = np.ogrid[:224, :224]
    arr[(X - 112)**2 + (Y - 112)**2 <= 60**2] = [255, 0, 0]
    test_img = Image.fromarray(arr)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    eng, sp_cls, info = make_fn(VL_MODEL, max_ns=8, num_blocks_override=580)
    sync()
    load_s = time.perf_counter() - t0
    print(f"  MM init={load_s:.2f}s  cg={info['cg']}  compile={info.get('compile', False)}",
          file=sys.stderr, flush=True)

    res = {"info": {"engine_init_s": round(load_s, 2),
                    "peak_mem_init_gb": round(peak_gb(), 2)},
           "multimodal": {}}

    cases = {"single_image": "Describe this image in detail.",
             "mixed": ("Here is an image. ") * 20 + " Describe it.",
             "pure_image": "What is in this image?",
             "text_only": "Explain the theory of relativity."}

    for case, prompt in cases.items():
        imgs = [test_img] if case != "text_only" else None
        data = []
        for i in range(12):
            first = None
            n = 0
            sync()
            t0g = time.perf_counter()
            for _ in eng.generate(prompt, max_new_tokens=64, temperature=0.0, images=imgs):
                n += 1
                if first is None:
                    sync()
                    first = time.perf_counter() - t0g
            sync()
            total = time.perf_counter() - t0g
            pt = len(eng.tokenizer.encode(prompt))
            d = pack(first or total, total, n, pt)
            if i >= 4:
                data.append(d)
        r = {"ttft_ms": round(median([d["ttft_s"] for d in data]) * 1000, 2),
             "decode_tok_s": round(mean([d["decode_tok_s"] for d in data]), 2),
             "e2e_tok_s": round(mean([d["e2e_tok_s"] for d in data]), 2)}
        res["multimodal"][case] = r
        print(f"  {case:<16s} TTFT={r['ttft_ms']:.1f}ms  "
              f"decode={r['decode_tok_s']:.1f}t/s",
              file=sys.stderr, flush=True)

    eng.destroy()
    del eng
    torch.cuda.empty_cache()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=["v4", "v5"], required=True)
    p.add_argument("--mode", choices=["text", "mm", "both"], default="text")
    p.add_argument("--out", default="")
    args = p.parse_args()

    ver = args.version
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    print(f"{ver} on {gpu} ({vram:.1f}GB)", file=sys.stderr, flush=True)

    if ver == "v4":
        make_fn, sfn, bfn = make_v4, bench_v4_single, bench_v4_batch
    else:
        make_fn, sfn, bfn = make_v5, bench_v5_single, bench_v5_batch

    result = {}
    if args.mode in ("text", "both"):
        result["text"] = run_text(make_fn, sfn, bfn)
    if args.mode in ("mm", "both"):
        result["mm"] = run_mm(make_fn, sfn, bfn)

    output = json.dumps(result, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
