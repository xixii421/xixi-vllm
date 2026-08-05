#!/usr/bin/env python3
"""v3 prefill 详细 profile：eager vs piecewise CUDA Graph。

用 torch.profiler 抓 prefill 全流程的 kernel 耗时，输出按 CUDA time 排序的
top kernel 表 + 按算子类别汇总表，用于定位 prefill 阶段的优化空间。

Usage（在 repo 根目录）：
  CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES=0 PYTHONPATH=vllm-v3 \
    python vllm-v3/bench/profile_prefill.py --mode eager --len 4096
  ... --mode cg --len 4096
  ... --model /home/xingyongwang/huggingface/Qwen3-4B --len 4096 --max-batched 4096
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

MODE_CAT = [
    ("GEMM", ("mm", "gemm", "gemv", "cublas", "sgemm", "hgemm", "dgemm", "trmm", "cublasLt")),
    ("Attention", ("flashinfer", "flash_infer", "prefill", "batch_prefill", "paged", "flash_attn", "qk", "pv", "rasterize")),
    ("Norm", ("rms", "norm", "welford", "residual", "layernorm", "fused_residual")),
    ("RoPE", ("rope", "rotary", "apply_rotary")),
    ("Activation", ("silu", "act", "gated", "relu", "gelu")),
    ("Elementwise", ("elementwise", "binary", "unary", "add_", "mul_", "copy", "where", "convert", "cast")),
    ("Sampling", ("sample", "multinomial", "gumbel", "argmax", "topk", "top_k")),
    ("Softmax", ("softmax", "masked_softmax", "exp")),
    ("KV-store", ("kv", "cache", "store", "writeback", "reshape")),
    ("Indexing", ("index", "gather", "scatter", "cat", "stack", "select", "copy_", "tensor")),
]


def categorize(name: str) -> str:
    low = name.lower()
    for cat, pats in MODE_CAT:
        for p in pats:
            if p in low:
                return cat
    return "Other"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["eager", "cg"], default="eager")
    p.add_argument("--len", type=int, default=4096, help="prompt 长度")
    p.add_argument("--model", default="/home/xingyongwang/huggingface/Qwen3-0.6B")
    p.add_argument("--max-batched", type=int, default=8192)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--iters", type=int, default=3, help="profile 的 prefill 次数")
    p.add_argument("--no-fa2", action="store_true", help="首 chunk prefill 强制用 FlashInfer（对照）")
    args = p.parse_args()

    sys_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # src 位于脚本上一级：vllm-v3/src
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src import Engine, EngineConfig, SamplingParams

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    eng = Engine(EngineConfig(
        model=args.model,
        context_len=16384,
        enforce_eager=(args.mode != "cg"),   # CG 模式需保留 decode dummy page
        prefill_cuda_graph=(args.mode == "cg"),
        dtype="auto",
        prefix_backend="none",
        max_num_seqs=8,
        max_num_batched_tokens=args.max_batched,
        gpu_memory_utilization=0.85,
    ))

    if args.no_fa2:
        _orig_build = eng.model_runner.backend.build_metadata

        def _no_fa2_build(*a, **k):
            k["use_fa2"] = False
            return _orig_build(*a, **k)

        eng.model_runner.backend.build_metadata = _no_fa2_build

    chunk = ("The quick brown fox jumps over the lazy dog. "
             "Explain continuous batching for LLM inference. ") * 4
    ids = []
    while len(ids) < args.len:
        ids.extend(eng.tokenizer.encode(chunk, add_special_tokens=False))
    ids = ids[:args.len]
    print(f"mode={args.mode} prompt_len={len(ids)} model={args.model.split('/')[-1]}",
          file=sys.stderr)

    # warmup：跑一次完整 prefill，避免首次分配污染
    def run_once() -> None:
        sp = SamplingParams(max_tokens=1, temperature=0.0)
        seq = eng.add_request(ids, sp)
        t0 = time.perf_counter()
        while not seq.is_finished:
            eng.step()
        return time.perf_counter() - t0

    run_once()
    torch.cuda.synchronize()

    # 正式 profile
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(args.iters):
            run_once()
            torch.cuda.synchronize()

    evs = prof.key_averages()
    cuda_total = sum(getattr(e, "self_device_time_total", 0) for e in evs) or 1.0

    print(f"\n{'=' * 100}", file=sys.stderr)
    print(f"  top {args.top} kernels by CUDA time  (total self_cuda={cuda_total/1000:.1f}ms for {args.iters} runs)", file=sys.stderr)
    print(f"{'=' * 100}", file=sys.stderr)
    h = (f"| {'kernel':<55s} | {'cat':<12s} | {'self_cuda':>10s} | {'%':>6s} | "
         f"{'calls':>6s} | {'avg_us':>8s} |")
    print(h, file=sys.stderr)
    print(f"| {'-'*55} | {'-'*12} | {'-'*10} | {'-'*6} | {'-'*6} | {'-'*8} |", file=sys.stderr)
    for e in sorted(evs, key=lambda x: getattr(x, "self_device_time_total", 0), reverse=True)[: args.top]:
        t = getattr(e, "self_device_time_total", 0)
        if t <= 0:
            continue
        cat = categorize(e.key)
        pct = 100.0 * t / cuda_total
        avg = t / max(1, e.count) / 1000  # us
        print(f"| {e.key[:55]:<55s} | {cat:<12s} | {t/1000:>9.1f}ms | "
              f"{pct:>5.1f}% | {e.count:>6d} | {avg:>7.1f} |", file=sys.stderr)

    # 按类别汇总
    print(f"\n{'=' * 100}", file=sys.stderr)
    print(f"  按算子类别汇总（self_cuda_time）", file=sys.stderr)
    print(f"{'=' * 100}", file=sys.stderr)
    cat_map: dict[str, tuple[float, int]] = {}
    for e in evs:
        c = categorize(e.key)
        t = getattr(e, "self_device_time_total", 0)
        ct, n = cat_map.get(c, (0.0, 0))
        cat_map[c] = (ct + t, n + e.count)
    print(f"| {'cat':<14s} | {'self_cuda':>10s} | {'%':>6s} | {'calls':>7s} |", file=sys.stderr)
    for c, (t, n) in sorted(cat_map.items(), key=lambda kv: -kv[1][0]):
        print(f"| {c:<14s} | {t/1000:>9.1f}ms | {100*t/cuda_total:>5.1f}% | {n:>7d} |", file=sys.stderr)

    # 每 run 的 wall 时间
    print(f"\n单次 prefill wall time ≈ {run_once()*1000:.1f} ms (3 次均值, 未含 profiler 开销)", file=sys.stderr)

    eng.destroy()


if __name__ == "__main__":
    main()
