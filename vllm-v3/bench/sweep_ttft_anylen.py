#!/usr/bin/env python3
"""任意 prompt 长度的 TTFT 扫描：v3 vs v4（prefill piecewise CUDA Graph）。

分析目标：
  1. 任意 token 长度下 TTFT 是否「均匀地」快 —— 关注 piece 网格造成的
     「锯齿」(sawtooth)：off-grid 长度被 pad 到下一个捕获 L，实际计算量
     = 完整 L 而非真实 N。
  2. N 超过 8192（或 max_num_batched_tokens）后的分块回退行为。
  3. 不同模型规模（0.6B/1.7B/4B/8B）下上述规律的缩放。

用法（在 repo 根目录）：
  # v3（prefill CG）
  CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES=0 PYTHONPATH=vllm-v3 \
    python vllm-v3/bench/sweep_ttft_anylen.py --backend v3 \
    --model /home/xingyongwang/huggingface/Qwen3-0.6B --out /tmp/sweep_v3_06b.json

  # v4
  CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES=0 PYTHONPATH=vllm-v4 \
    python vllm-v3/bench/sweep_ttft_anylen.py --backend v4 \
    --model /home/xingyongwang/huggingface/Qwen3-0.6B --out /tmp/sweep_v4_06b.json

  # v3 对照：关掉 prefill CG（eager）
  ... --backend v3 --no-prefill-cg --out /tmp/sweep_v3eager_06b.json
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

DEFAULT_LENS = [
    128, 192, 256, 300, 384, 512, 600, 700, 768, 900, 1024,
    1100, 1280, 1500, 1600, 1792, 2048, 2500, 3000, 3500, 4000,
    4096, 5000, 6000, 7000, 8000, 8192, 9000, 10000, 12000, 14000,
]

# 网格上的「标准」长度（正好等于捕获 L 的点），用于对比锯齿
GRID_LENS = [128, 256, 512, 1024, 2048, 4096, 8192]


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2] if s else 0.0


def make_prompt_tokens(tokenizer, target_len: int) -> list[int]:
    chunk = (
        "The quick brown fox jumps over the lazy dog. "
        "Explain continuous batching for LLM inference. "
    ) * 4
    ids: list[int] = []
    while len(ids) < target_len:
        ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
    return ids[:target_len]


def build_engine(backend: str, model: str, prefill_cg: bool, gpu_util: float, kv_blocks: int, max_batched: int) -> tuple:
    """返回 (Engine, SamplingParams 类, tag)。"""
    from src import Engine, EngineConfig, SamplingParams

    cfg = EngineConfig(
        model=model,
        context_len=16384,
        enforce_eager=False,
        prefill_cuda_graph=prefill_cg,
        dtype="auto",
        prefix_backend="none",
        max_num_seqs=8,
        max_num_batched_tokens=max_batched,
        gpu_memory_utilization=gpu_util,
        num_kvcache_blocks=kv_blocks,
    )
    eng = Engine(cfg)
    tag = f"{backend}-prefillCG" if prefill_cg else f"{backend}-eager"
    return eng, SamplingParams, tag


def piece_grid(eng) -> list[int]:
    """读取已捕获的 prefill CG piece 的 L 网格（单 seq 可用的部分）。"""
    try:
        runner = eng.model_runner.prefill_cg_runner
        if runner is None:
            return []
        pieces = runner.pieces
        # 单 seq(bs=1)：可用 L = 所有 (1, L) piece 的 L
        ls = sorted({L for (bs, L) in pieces if bs == 1})
        return ls
    except Exception:
        return []


def measure_ttft(eng, sp_cls, prompt_ids: list[int]) -> float:
    """单个 prefill 的 TTFT（秒），gen=1 token。"""
    sp = sp_cls(max_tokens=1, temperature=0.0)
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["v3", "v4"], required=True)
    p.add_argument("--model", default="/home/xingyongwang/huggingface/Qwen3-0.6B")
    p.add_argument("--lens", default=",".join(map(str, DEFAULT_LENS)))
    p.add_argument("--out", required=True)
    p.add_argument("--no-prefill-cg", action="store_true", help="关掉 prefill CG（eager 对照）")
    p.add_argument("--gpu-util", type=float, default=0.85,
                   help="gpu_memory_utilization；大模型 prefill CG 捕获需要更多显存余量，建议 4B=0.5 / 8B=0.45")
    p.add_argument("--kv-blocks", type=int, default=0,
                   help="固定 KV 块数（0=自动）。大模型配低 gpu-util 时 KV pool 会被算小，"
                        "单文本扫描建议显式给足（4B=100, 8B=60）")
    p.add_argument("--max-batched", type=int, default=8192,
                   help="max_num_batched_tokens；8B 在 24GB 上需降到 2048 才能捕获 prefill CG")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=6)
    args = p.parse_args()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    lens = [int(x) for x in args.lens.split(",") if x.strip()]
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}", file=sys.stderr)
    print(f"backend={args.backend} prefill_cg={not args.no_prefill_cg} gpu_util={args.gpu_util} kv_blocks={args.kv_blocks} max_batched={args.max_batched}", file=sys.stderr)

    t0 = time.perf_counter()
    eng, sp_cls, tag = build_engine(args.backend, args.model, not args.no_prefill_cg, args.gpu_util, args.kv_blocks, args.max_batched)
    sync()
    init_s = time.perf_counter() - t0
    grid = piece_grid(eng)
    print(f"Init: {init_s:.2f}s  mode={tag}", file=sys.stderr)
    print(f"Prefill CG L-grid (bs=1): {grid}", file=sys.stderr)

    results = {
        "backend": args.backend,
        "mode": tag,
        "model": args.model,
        "gpu": gpu,
        "init_s": round(init_s, 2),
        "piece_ls": grid,
        "context_len": eng.context_len,
        "sweep": {},
    }

    def _predict_piece(N: int) -> int:
        for L in grid:
            if L >= N:
                return L
        return -1  # 超出网格 → 回退 eager

    for plen in lens:
        tokens = make_prompt_tokens(eng.tokenizer, plen)
        n = len(tokens)
        if n > eng.context_len:
            print(f"  len={n:>5d}: SKIP (> context_len={eng.context_len})", file=sys.stderr)
            results["sweep"][str(n)] = {"prompt_len": n, "skipped": "over_context"}
            continue

        print(f"  len={n:>5d} -> piece L={_predict_piece(n):>5d}: ", file=sys.stderr, end="", flush=True)
        ttfts: list[float] = []
        for i in range(args.warmup + args.iters):
            ttft = measure_ttft(eng, sp_cls, tokens)
            if i >= args.warmup:
                ttfts.append(ttft)

        r = {
            "prompt_len": n,
            "piece_L": _predict_piece(n),
            "on_grid": n in grid,
            "ttft_ms_mean": round(mean(ttfts) * 1000, 2),
            "ttft_ms_median": round(median(ttfts) * 1000, 2),
            "ttft_ms_min": round(min(ttfts) * 1000, 2),
            "prefill_tok_s": round(n / mean(ttfts), 1),
            "waste_ratio": round(_predict_piece(n) / n, 3),
        }
        results["sweep"][str(n)] = r
        print(f"TTFT={r['ttft_ms_median']:.2f}ms  tok/s={r['prefill_tok_s']:.0f}", file=sys.stderr, flush=True)

    # 表格
    print(f"\n{'=' * 78}", file=sys.stderr)
    print(f"  {tag}  ({args.model.split('/')[-1]})", file=sys.stderr)
    print(f"{'=' * 78}", file=sys.stderr)
    h = f"| {'N':>6s} | {'piece':>6s} | {'waste':>6s} | {'TTFT(med)':>9s} | {'tok/s':>7s} |"
    print(h, file=sys.stderr)
    print(f"| {'-'*6} | {'-'*6} | {'-'*6} | {'-'*9} | {'-'*7} |", file=sys.stderr)
    for plen in lens:
        key = str(min(plen, eng.context_len))
        if key not in results["sweep"]:
            continue
        r = results["sweep"][key]
        if r.get("skipped"):
            print(f"| {r['prompt_len']:>6d} |  SKIP  |        |           |         |", file=sys.stderr)
            continue
        print(
            f"| {r['prompt_len']:>6d} | {r['piece_L']:>6d} | {r['waste_ratio']:>5.2f}x | "
            f"{r['ttft_ms_median']:>7.2f}ms | {r['prefill_tok_s']:>6.0f} |",
            file=sys.stderr,
        )

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.out}", file=sys.stderr)

    eng.destroy()
    del eng
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
