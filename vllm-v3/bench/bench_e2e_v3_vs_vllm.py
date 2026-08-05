#!/usr/bin/env python3
"""vllm-v3 vs 官方 vLLM (0.26.0) 端到端对比。

同一模型 Qwen3-0.6B、同一 GPU、同一 token 级 prompt，对比三类指标：
  1. TTFT（单请求）：prompt ∈ {128,256,512,1024,2048}，gen=1
  2. 单请求 E2E：prompt=256，gen=128 → TTFT + decode tok/s + e2e tok/s
  3. 并发吞吐：bs ∈ {4,8}，prompt=256，gen=128/req → 聚合 tok/s + 平均 TTFT + e2e wall

v3 测两种配置（eager / cg），官方 vLLM 用默认最优配置（CUDA Graph 开）。
公平性：同一 context_len(8192)、同一 gpu_memory_utilization(0.85)、temperature=0。

用法:
  # v3（任意配置）
  PYTHONPATH=vllm-v3 .venv2/bin/python vllm-v3/bench/bench_e2e_v3_vs_vllm.py --backend v3 --cfg cg --out /tmp/e2e_v3_cg.json
  # 官方 vLLM
  .venv-vllm/bin/python vllm-v3/bench/bench_e2e_v3_vs_vllm.py --backend vllm --out /tmp/e2e_vllm.json
  # 一键跑全部（串行，各自解释器）
  python vllm-v3/bench/bench_e2e_v3_vs_vllm.py --backend both --out /tmp/e2e_all.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # 4090 D：两引擎同卡对比

import torch
from transformers import AutoTokenizer

MODEL = "/home/xingyongwang/huggingface/Qwen3-0.6B"
CTX_LEN = 8192
GPU_UTIL = 0.85
TTFT_LENS = [128, 256, 512, 1024, 2048]
E2E_PROMPT = 256
E2E_GEN = 128
CONC_GEN = 128
BATCH_SIZES = [4, 8]
WARMUP = 2
ITERS = 5

BASE = ("The quick brown fox jumps over the lazy dog. "
        "Explain continuous batching for LLM inference in clear prose. "
        "Cover scheduling, KV cache paging, and why decode throughput scales "
        "with concurrency. Keep the answer factual and structured. ")


def sync() -> None:
    torch.cuda.synchronize()


def make_prompt_ids(tok: AutoTokenizer, n: int) -> list[int]:
    ids: list[int] = []
    while len(ids) < n:
        ids.extend(tok.encode(BASE, add_special_tokens=False))
    return ids[:n]


# ──────────────────────────────────────────────────────────────────────
#  vllm-v3 backend
# ──────────────────────────────────────────────────────────────────────

V3_CFGS = {
    "eager": dict(enforce_eager=True, prefill_cuda_graph=False),
    "cg":    dict(enforce_eager=False, prefill_cuda_graph=True),
}


def bench_v3(tok, cfg_name: str) -> dict:
    from src import Engine, EngineConfig, SamplingParams

    eng = Engine(EngineConfig(
        model=MODEL, context_len=CTX_LEN, dtype="auto", prefix_backend="none",
        max_num_seqs=16, max_num_batched_tokens=8192,
        num_kvcache_blocks=0, gpu_memory_utilization=GPU_UTIL,
        **V3_CFGS[cfg_name],
    ))
    res: dict = {"cfg": cfg_name}

    # 1) TTFT 单请求
    ttft: dict[str, float] = {}
    for n in TTFT_LENS:
        pids = make_prompt_ids(tok, n)
        vals = []
        for _ in range(WARMUP + ITERS):
            seq = eng.add_request(pids, SamplingParams(max_tokens=1, temperature=0.0))
            sync()
            t0 = time.perf_counter()
            first = None
            while not seq.is_finished:
                eng.step()
                if first is None and seq.num_completion_tokens > 0:
                    sync()
                    first = (time.perf_counter() - t0) * 1000
            sync()
            vals.append(first if first is not None else (time.perf_counter() - t0) * 1000)
        ttft[str(n)] = round(statistics.median(vals[WARMUP:]), 3)
    res["ttft_ms"] = ttft

    # 2) 单请求 E2E（gen=128）
    pids = make_prompt_ids(tok, E2E_PROMPT)
    e2e_rows = []
    for _ in range(WARMUP + ITERS):
        seq = eng.add_request(pids, SamplingParams(max_tokens=E2E_GEN, temperature=0.0))
        sync()
        t0 = time.perf_counter()
        first = None
        while not seq.is_finished:
            eng.step()
            if first is None and seq.num_completion_tokens > 0:
                sync()
                first = (time.perf_counter() - t0) * 1000
        sync()
        total = (time.perf_counter() - t0) * 1000
        e2e_rows.append({"ttft": first, "total": total, "gen": seq.num_completion_tokens})
    rows = e2e_rows[WARMUP:]
    ttft_med = statistics.median(r["ttft"] for r in rows)
    total_med = statistics.median(r["total"] for r in rows)
    gen = rows[0]["gen"]
    res["e2e_single"] = {
        "ttft_ms": round(ttft_med, 3),
        "e2e_ms": round(total_med, 3),
        "decode_tok_s": round((gen - 1) / (total_med - ttft_med) * 1000, 2) if total_med > ttft_med else 0,
        "e2e_tok_s": round(gen / total_med * 1000, 2),
    }

    # 3) 并发吞吐
    res["concurrency"] = {}
    for bs in BATCH_SIZES:
        pids = [make_prompt_ids(tok, E2E_PROMPT) for _ in range(bs)]
        rates, walls, ttfts = [], [], []
        for _ in range(WARMUP + ITERS):
            seqs = [eng.add_request(p, SamplingParams(max_tokens=CONC_GEN, temperature=0.0))
                    for p in pids]
            sync()
            t0 = time.perf_counter()
            first_at = {}
            while not eng.scheduler.is_finished():
                eng.step()
                for s in seqs:
                    if s.seq_id not in first_at and s.num_completion_tokens > 0:
                        first_at[s.seq_id] = (time.perf_counter() - t0) * 1000
            sync()
            wall = (time.perf_counter() - t0) * 1000
            tot = sum(s.num_completion_tokens for s in seqs)
            rates.append(tot / wall * 1000)
            walls.append(wall)
            ttfts.extend(first_at.values())
        res["concurrency"][f"bs{bs}"] = {
            "tok_s": round(statistics.mean(rates), 2),
            "wall_ms": round(statistics.median(walls), 3),
            "ttft_mean_ms": round(statistics.mean(ttfts), 3),
        }

    eng.destroy()
    gc.collect()
    torch.cuda.empty_cache()
    return res


# ──────────────────────────────────────────────────────────────────────
#  official vLLM backend
# ──────────────────────────────────────────────────────────────────────

def bench_vllm(tok) -> dict:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        max_model_len=CTX_LEN,
        gpu_memory_utilization=GPU_UTIL,
        trust_remote_code=True,
        disable_log_stats=False,  # 关闭后 RequestOutput.metrics 为 None，无法取 TTFT
    )
    res: dict = {"cfg": "vllm_default"}

    def _run(pids_list, gen) -> tuple[list, float]:
        sp = SamplingParams(max_tokens=gen, temperature=0.0)
        t0 = time.monotonic()
        outs = llm.generate(
            prompts=pids_list,
            sampling_params=sp,
            use_tqdm=False,
        )
        return outs, time.monotonic() - t0

    # 1) TTFT 单请求（gen=1）
    ttft: dict[str, float] = {}
    for n in TTFT_LENS:
        pids = [make_prompt_ids(tok, n)]
        vals = []
        for _ in range(WARMUP + ITERS):
            outs, _ = _run(pids, 1)
            m = outs[0].metrics
            # first_token_latency 是单调时钟差值（含排队），比手工 wall 更准
            vals.append((m.first_token_latency if m and m.first_token_latency else 0.0) * 1000)
        ttft[str(n)] = round(statistics.median(vals[WARMUP:]), 3)
    res["ttft_ms"] = ttft

    # 2) 单请求 E2E（gen=128）：total 用 wall（monotonic），TTFT 用 metrics
    pids = [make_prompt_ids(tok, E2E_PROMPT)]
    e2e_rows = []
    for _ in range(WARMUP + ITERS):
        sync()
        t_wall = time.monotonic()
        outs, _ = _run(pids, E2E_GEN)
        sync()
        wall = (time.monotonic() - t_wall) * 1000
        m = outs[0].metrics
        e2e_rows.append({
            "ttft": (m.first_token_latency if m and m.first_token_latency else 0.0) * 1000,
            "total": wall,
            "gen": len(outs[0].outputs[0].token_ids),
        })
    rows = e2e_rows[WARMUP:]
    ttft_med = statistics.median(r["ttft"] for r in rows)
    total_med = statistics.median(r["total"] for r in rows)
    gen_n = rows[0]["gen"]
    res["e2e_single"] = {
        "ttft_ms": round(ttft_med, 3),
        "e2e_ms": round(total_med, 3),
        "decode_tok_s": round((gen_n - 1) / (total_med - ttft_med) * 1000, 2) if total_med > ttft_med else 0,
        "e2e_tok_s": round(gen_n / total_med * 1000, 2),
    }

    # 3) 并发吞吐
    res["concurrency"] = {}
    for bs in BATCH_SIZES:
        pids = [make_prompt_ids(tok, E2E_PROMPT) for _ in range(bs)]
        rates, walls, ttfts = [], [], []
        for _ in range(WARMUP + ITERS):
            sync()
            t0 = time.perf_counter()
            outs, _ = _run(pids, CONC_GEN)
            sync()
            wall = (time.perf_counter() - t0) * 1000
            tot = sum(len(o.outputs[0].token_ids) for o in outs)
            rates.append(tot / wall * 1000)
            walls.append(wall)
            for o in outs:
                if o.metrics and o.metrics.first_token_latency:
                    ttfts.append(o.metrics.first_token_latency * 1000)
        res["concurrency"][f"bs{bs}"] = {
            "tok_s": round(statistics.mean(rates), 2),
            "wall_ms": round(statistics.median(walls), 3),
            "ttft_mean_ms": round(statistics.mean(ttfts), 3),
        }

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return res


# ──────────────────────────────────────────────────────────────────────
#  main / orchestration
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["v3", "vllm", "both"], default="both")
    p.add_argument("--cfg", default="all", choices=list(V3_CFGS) + ["all"])
    p.add_argument("--out", default="/tmp/e2e_all.json")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    results: dict = {"meta": {"model": MODEL, "context_len": CTX_LEN, "gpu_util": GPU_UTIL,
                              "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")}}

    if args.backend in ("v3", "both"):
        results["v3"] = {}
        cfgs = list(V3_CFGS) if args.cfg == "all" else [args.cfg]
        for cfg in cfgs:
            print(f"\n=== v3 [{cfg}] ===", flush=True)
            results["v3"][cfg] = bench_v3(tok, cfg)
            print(json.dumps(results["v3"][cfg], indent=1))

    if args.backend in ("vllm", "both"):
        print("\n=== vllm (official 0.26) ===", flush=True)
        results["vllm"] = bench_vllm(tok)
        print(json.dumps(results["vllm"], indent=1))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
