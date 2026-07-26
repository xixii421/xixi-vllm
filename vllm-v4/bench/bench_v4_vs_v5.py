#!/usr/bin/env python3
"""Comprehensive performance comparison: vllm-v4 vs vllm-v5.

Measures TTFT, decode tok/s, E2E throughput across single/multi-request
text and multimodal workloads. Both engines at max-performance settings.

Usage:
  PYTHONPATH=/home/xingyongwang/code/mycuda/dzyy-vllm/vllm-v4 \\
    python bench/bench_v4_vs_v5.py --mode both --out results_v4v5_all.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

TEXT_MODEL = "/home/xingyongwang/huggingface/Qwen3-0.6B"
VL_MODEL   = "/home/xingyongwang/huggingface/Qwen3-VL-2B-Instruct"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mean(xs):
    return float(statistics.mean(xs)) if xs else 0.0


def median(xs):
    return float(statistics.median(xs)) if xs else 0.0


def peak_gb():
    return torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0


NON_BLOCKING_PROMPT = (
    "Explain continuous batching for LLM inference in clear prose. "
    "Cover scheduling, KV cache paging, and why decode throughput scales with "
    "concurrency. Keep the answer factual and structured. "
) * 8

# ===========================================================================
#  Engine factory + per-version benchmark functions
# ===========================================================================


def make_v4(model_path, max_num_seqs=8, gpu_util=0.85, context_len=4096):
    from src.config import EngineConfig
    from src.engine.engine import Engine
    from src.sampling_params import SamplingParams

    cfg = EngineConfig(
        model=model_path,
        context_len=context_len,
        enforce_eager=False,
        dtype="float16",
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=gpu_util,
        mix_prefill_decode=True,
        compile_prefill=True,
        prefix_backend="none",
    )
    eng = Engine(cfg)
    info = {
        "cg_enabled": eng.model_runner.cuda_graph_runner is not None,
        "prefill_compiled": eng.model_runner._prefill_model is not None,
    }
    return eng, SamplingParams, info


def bench_v4_single(eng, sp_cls, prompt, max_new):
    sp = sp_cls(max_tokens=max_new, temperature=0.0)
    seq = eng.add_request(prompt, sp)
    first = None
    n = 0
    sync()
    t0 = time.perf_counter()
    while not seq.is_finished:
        eng.step()
        if first is None and seq.num_completion_tokens > 0:
            sync()
            first = time.perf_counter() - t0
    sync()
    total = time.perf_counter() - t0
    pt = len(eng.tokenizer.encode(prompt))
    return _pack(first or total, total, n, pt)


def bench_v4_batch(eng, sp_cls, prompt, max_new, bs):
    sp = sp_cls(max_tokens=max_new, temperature=0.0)
    seqs = [eng.add_request(prompt, sp) for _ in range(bs)]
    sync()
    t0 = time.perf_counter()
    while not all(s.is_finished for s in seqs):
        eng.step()
    sync()
    total = time.perf_counter() - t0
    gens = sum(s.num_completion_tokens for s in seqs)
    pt = len(eng.tokenizer.encode(prompt))
    return _pack(total, total, gens, pt, note=f"batch{bs}")


def make_v5(model_path, max_num_seqs=8, gpu_util=0.85, context_len=4096):
    from src import Engine, EngineConfig, SamplingParams

    cfg = EngineConfig(
        model=model_path,
        context_len=context_len,
        enforce_eager=False,
        dtype="float16",
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=gpu_util,
        enable_radix=False,
        tp_size=1, pp_size=1,
    )
    eng = Engine(cfg)
    info = {"cg_enabled": True, "prefill_compiled": False}
    return eng, SamplingParams, info


def bench_v5_single(eng, sp_cls, prompt, max_new):
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
    pt = len(eng.tokenizer.encode(prompt))
    return _pack(first or total, total, n, pt)


def bench_v5_batch(eng, sp_cls, prompt, max_new, bs):
    sp = sp_cls(max_tokens=max_new, temperature=0.0)
    sync()
    t0 = time.perf_counter()
    out = eng.generate_batch([prompt] * bs, sp)
    sync()
    total = time.perf_counter() - t0
    gens = sum(len(eng.tokenizer.encode(t)) for t in out)
    pt = len(eng.tokenizer.encode(prompt))
    return _pack(total, total, gens, pt, note=f"batch{bs}")


def _pack(ttft, total, gen_n, prompt_n, note=""):
    if total > ttft * 1.02 and gen_n > 1:
        decode_s = total - ttft
        decode_tok_s = (gen_n - 1) / decode_s
    else:
        decode_s = total
        decode_tok_s = gen_n / total if total > 0 else 0.0
    return {
        "ttft_s": round(ttft, 4),
        "total_s": round(total, 4),
        "decode_tok_s": round(decode_tok_s, 2),
        "e2e_tok_s": round(gen_n / total, 2) if total > 0 else 0.0,
        "e2e_latency_s": round(total, 4),
        "gen_tokens": gen_n,
        "prompt_tokens": prompt_n,
        "note": note,
    }


# ===========================================================================
#  Multimodal
# ===========================================================================

def bench_mm(eng, prompt, images, max_new):
    vit_s = 0.0
    if hasattr(eng.model, "encode_and_merge"):
        orig = eng.model.encode_and_merge

        def _timed(*a, **kw):
            nonlocal vit_s
            sync()
            t0v = time.perf_counter()
            out = orig(*a, **kw)
            sync()
            vit_s += time.perf_counter() - t0v
            return out
        eng.model.encode_and_merge = _timed

    first = None
    n = 0
    sync()
    t0 = time.perf_counter()
    for _ in eng.generate(prompt, max_new_tokens=max_new, temperature=0.0, images=images):
        n += 1
        if first is None:
            sync()
            first = time.perf_counter() - t0
    sync()
    total = time.perf_counter() - t0
    pt = len(eng.tokenizer.encode(prompt))
    d = _pack(first or total, total, n, pt)
    d["vit_encode_s"] = round(vit_s, 4)
    return d


# ===========================================================================
#  Runner
# ===========================================================================

def run_version(version, make_fn, single_fn, batch_fn, args):
    print(f"\n--- {version} ---")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    eng, sp_cls, info = make_fn(
        args.model,
        max_num_seqs=max(args.batch_sizes) + 2,
        gpu_util=args.gpu_util,
        context_len=args.context_len,
    )
    sync()
    load_s = time.perf_counter() - t0
    print(f"  Init: {load_s:.2f}s  CG={info['cg_enabled']} compile={info['prefill_compiled']}")

    res = {"engine_init_s": round(load_s, 2), "peak_mem_init_gb": round(peak_gb(), 2),
           "cg_enabled": info["cg_enabled"], "prefill_compiled": info["prefill_compiled"]}

    prompt = args.prompt

    # Single-request sweep
    res["single"] = {}
    for label, nt in [("TTFT_1tok", 1), ("short_32tok", 32),
                      ("med_128tok", 128), ("long_256tok", 256)]:
        ttfts, totals = [], []
        for i in range(args.warmup + args.iters):
            d = single_fn(eng, sp_cls, prompt, nt)
            if i >= args.warmup:
                ttfts.append(d["ttft_s"])
                totals.append(d["total_s"])
        r = {
            "ttft_ms": round(median(ttfts) * 1000, 2),
            "e2e_s": round(mean(totals), 4),
            "decode_tok_s": round((nt - 1) / (mean(totals) - mean(ttfts)), 2)
            if mean(totals) > mean(ttfts) * 1.02 and nt > 1 else 0,
            "e2e_tok_s": round(nt / mean(totals), 2) if mean(totals) > 0 else 0,
        }
        res["single"][label] = r
        print(f"  {label:<18s}  TTFT={r['ttft_ms']:.1f}ms  "
              f"decode={r['decode_tok_s']:.1f}t/s  e2e={r['e2e_tok_s']:.1f}t/s")

    # Concurrency sweep
    res["concurrency"] = {}
    for bs in args.batch_sizes:
        rates, walls = [], []
        w = max(3, args.warmup)
        for i in range(w + args.iters):
            d = batch_fn(eng, sp_cls, prompt, args.max_new, bs)
            if i >= w:
                rates.append(d["e2e_tok_s"])
                walls.append(d["e2e_latency_s"])
        r = {
            "e2e_tok_s": round(mean(rates), 2),
            "e2e_latency_s": round(mean(walls), 4),
            "peak_mem_gb": round(peak_gb(), 2),
        }
        res["concurrency"][f"bs{bs}"] = r
        print(f"  bs={bs:<3d}  throughput={r['e2e_tok_s']:.1f}t/s  "
              f"latency={r['e2e_latency_s']*1000:.0f}ms  mem={r['peak_mem_gb']:.1f}GB")

    eng.destroy()
    del eng
    torch.cuda.empty_cache()
    return res


def run_mm_version(version, make_fn, args):
    print(f"\n--- {version} MM ---")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    eng, sp_cls, info = make_fn(
        args.mm_model,
        max_num_seqs=8,
        gpu_util=args.gpu_util,
        context_len=args.context_len,
    )
    sync()
    load_s = time.perf_counter() - t0
    print(f"  Init: {load_s:.2f}s  CG={info['cg_enabled']}")

    res = {"engine_init_s": round(load_s, 2), "peak_mem_init_gb": round(peak_gb(), 2)}

    from PIL import Image
    import numpy as np
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    Y, X = np.ogrid[:224, :224]
    arr[(X - 112)**2 + (Y - 112)**2 <= 60**2] = [255, 0, 0]
    test_img = Image.fromarray(arr)

    cases = {
        "single_image": "Describe this image in detail.",
        "mixed": ("Here is an image for you to analyze. Look at it carefully "
                  "and tell me what you see. ") * 20 + " Describe this image.",
        "pure_image": "What is in this image?",
        "text_only": "Explain the theory of relativity in simple terms.",
    }

    res["multimodal"] = {}
    for case, prompt in cases.items():
        imgs = [test_img] if case != "text_only" else None
        data = []
        w = max(3, args.warmup)
        for i in range(w + args.iters):
            d = bench_mm(eng, prompt, imgs, args.max_new)
            if i >= w:
                data.append(d)

        r = {
            "ttft_ms": round(median([d["ttft_s"] for d in data]) * 1000, 2),
            "vit_ms": round(mean([d.get("vit_encode_s", 0) for d in data]) * 1000, 2),
            "decode_tok_s": round(mean([d["decode_tok_s"] for d in data]), 2),
            "e2e_tok_s": round(mean([d["e2e_tok_s"] for d in data]), 2),
            "e2e_latency_s": round(mean([d["total_s"] for d in data]), 4),
        }
        res["multimodal"][case] = r
        vit = f"  vit={r['vit_ms']:.0f}ms" if r["vit_ms"] > 0 else ""
        print(f"  [{case:<15s}] TTFT={r['ttft_ms']:.1f}ms{vit}  "
              f"decode={r['decode_tok_s']:.1f}t/s  e2e={r['e2e_tok_s']:.1f}t/s")

    eng.destroy()
    del eng
    torch.cuda.empty_cache()
    return res


def print_comparison(title, v4, v5, keys):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"| {'Metric':<28s} | {'v4':>12s} | {'v5':>12s} | {'v4/v5':>8s} |")
    print(f"| {'-'*28} | {'-'*12} | {'-'*12} | {'-'*8} |")
    for k in keys:
        v4v = _nested_get(v4, k)
        v5v = _nested_get(v5, k)
        if isinstance(v4v, str) or v4v is None:
            continue
        if isinstance(v5v, str) or v5v is None:
            continue
        # ratio: bigger=better for tok_s, smaller=better for time/ms
        if "tok_s" in k:
            ratio = v4v / v5v if v5v else 0
        else:
            ratio = v5v / v4v if v4v else 0
        fmt = lambda x: f"{x*1000:.1f}ms" if abs(x) < 1 else f"{x:.2f}t/s" if "tok" in k else f"{x:.1f}"
        print(f"| {k.split('.')[-1]:<28s} | {v4v:>12.2f} | {v5v:>12.2f} | {ratio:>7.2f}× |")


def _nested_get(d, key):
    for k in key.split("."):
        if isinstance(d, dict):
            d = d.get(k, None)
        else:
            return None
    return d


# ===========================================================================
#  Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["text", "mm", "both"], default="text")
    p.add_argument("--model", default=TEXT_MODEL)
    p.add_argument("--mm-model", default=VL_MODEL)
    p.add_argument("--prompt", default=NON_BLOCKING_PROMPT)
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--context-len", type=int, default=4096)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--gpu-util", type=float, default=0.85)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--out", default="results_v4v5.json")
    args = p.parse_args()

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    print(f"GPU: {gpu}  VRAM: {vram:.1f}GB")

    all_results = {}

    if args.mode in ("text", "both"):
        rv4 = run_version("v4", make_v4, bench_v4_single, bench_v4_batch, args)
        rv5 = run_version("v5", make_v5, bench_v5_single, bench_v5_batch, args)
        all_results["text"] = {"v4": rv4, "v5": rv5}

        print("\n" + "=" * 75)
        print("  TEXT COMPARISON SUMMARY")
        print("=" * 75)
        print_comparison("TTFT (1 tok)", rv4["single"], rv5["single"],
                         ["TTFT_1tok.ttft_ms", "TTFT_1tok.e2e_tok_s"])
        print_comparison("Decode (256 tok)", rv4["single"], rv5["single"],
                         ["long_256tok.decode_tok_s", "long_256tok.e2e_tok_s"])
        for bs in args.batch_sizes:
            k = f"bs{bs}"
            print_comparison(f"Concurrency bs={bs}", rv4["concurrency"], rv5["concurrency"],
                             [f"{k}.e2e_tok_s", f"{k}.e2e_latency_s", f"{k}.peak_mem_gb"])
        print_comparison("Engine Init", rv4, rv5,
                         ["engine_init_s", "peak_mem_init_gb"])

    if args.mode in ("mm", "both"):
        torch.cuda.empty_cache()
        rv4 = run_mm_version("v4", make_v4, args)
        rv5 = run_mm_version("v5", make_v5, args)
        all_results["mm"] = {"v4": rv4, "v5": rv5}

        print("\n" + "=" * 75)
        print("  MULTIMODAL COMPARISON SUMMARY")
        print("=" * 75)
        for case in ["single_image", "mixed", "pure_image", "text_only"]:
            print_comparison(f"MM {case}", rv4["multimodal"], rv5["multimodal"],
                             [f"{case}.ttft_ms", f"{case}.decode_tok_s",
                              f"{case}.e2e_tok_s", f"{case}.vit_ms"])

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = RESULTS_DIR / out_path
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
