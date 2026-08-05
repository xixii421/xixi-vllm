#!/usr/bin/env python3
"""多并发下 CG 收益基准（v3）。

Part A — 纯 prefill 并发：所有请求同时到达、max_tokens=1（单步 prefill）。
  遍历并发度 bs ∈ {1,2,4,8} 与文本长度 len ∈ {129,513}，测整批 wall time 与
  每请求 TTFT，对比 eager / cg(prefill piecewise + decode CG)。
  关键观察：cg_max_tokens=1536 下 bs>1 piece 覆盖墙（bs=4 最长 L=256、bs=8 仅 L=128），
  bs=8 / 长文本会回退 eager —— 量化「覆盖缺口」。

Part B — prefill+decode 混合并发（交错到达）：
  第一波 len=129, max_tokens=32 的请求先进 decode，第 k 步追加第二波
  max_tokens=1 的 prefill。测端到端完成时间，对比 mix_prefill_decode 开/关、
  CG 开/关。（对应 v4 迁移到 v3 的「decode 优先 + 同一步补 prefill chunk」方案）

用法:
  PYTHONPATH=vllm-v3 .venv2/bin/python vllm-v3/bench/bench_concurrent.py [--part A|B]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from src import Engine, EngineConfig, SamplingParams

MODEL = "/home/xingyongwang/huggingface/Qwen3-0.6B"


def build_prompt(tok: AutoTokenizer, n: int) -> list[int]:
    chunk = ("The quick brown fox jumps over the lazy dog. "
             "Explain continuous batching for LLM inference. ") * 4
    ids: list[int] = []
    while len(ids) < n:
        ids.extend(tok.encode(chunk, add_special_tokens=False))
    return ids[:n]


def make_engine(**kw) -> Engine:
    return Engine(EngineConfig(
        model=MODEL, context_len=16384, dtype="auto", prefix_backend="none",
        max_num_seqs=16, num_kvcache_blocks=128, gpu_memory_utilization=0.85, **kw,
    ))


def teardown(eng: Engine | None) -> None:
    if eng is not None:
        eng.destroy()
    gc.collect()
    torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────
#  Part A: 纯 prefill 并发
# ──────────────────────────────────────────────────────────────────────

def bench_pure_prefill(tok, *, bs: int, plen: int, mode: str,
                       reps: int = 5) -> dict:
    """bs 个同长文本同时到达，max_tokens=1。返回批 wall time 与 TTFT 统计。"""
    cfg = {
        "eager": dict(enforce_eager=True, prefill_cuda_graph=False),
        "cg":    dict(enforce_eager=False, prefill_cuda_graph=True),
    }[mode]
    eng = make_engine(**cfg)
    prompts = [build_prompt(tok, plen) for _ in range(bs)]

    # 预热一次（触发 kernel/autotune/CG 已就绪）
    seqs = [eng.add_request(p, SamplingParams(max_tokens=1, temperature=0.0)) for p in prompts]
    while not eng.scheduler.is_finished():
        eng.step()
    teardown(eng)
    eng = None

    eng = make_engine(**cfg)
    walls: list[float] = []
    ttfts: list[float] = []
    steps_used = 0
    for _ in range(reps):
        seqs = [eng.add_request(p, SamplingParams(max_tokens=1, temperature=0.0)) for p in prompts]
        first_at: dict[int, float] = {}
        t0 = time.perf_counter()
        nsteps = 0
        while not eng.scheduler.is_finished():
            t_step = time.perf_counter()
            eng.step()
            nsteps += 1
            for s in seqs:
                if s.seq_id not in first_at and s.num_completion_tokens > 0:
                    first_at[s.seq_id] = (time.perf_counter() - t0) * 1000
        wall_ms = (time.perf_counter() - t0) * 1000
        walls.append(wall_ms)
        ttfts.extend(first_at.values())
        steps_used = max(steps_used, nsteps)

    # 报告本 mode 是否实际用了 prefill CG（看 pieces 命中情况）
    used_cg = False
    if eng.model_runner.prefill_cg_runner is not None:
        r = eng.model_runner.prefill_cg_runner
        piece = r.can_use(bs, [plen] * bs)
        used_cg = piece is not None
    teardown(eng)

    return {
        "mode": mode, "bs": bs, "plen": plen,
        "wall_mean_ms": round(statistics.mean(walls), 3),
        "wall_min_ms": round(min(walls), 3),
        "ttft_mean_ms": round(statistics.mean(ttfts), 3),
        "steps": steps_used,
        "cg_used": used_cg,
    }


# ──────────────────────────────────────────────────────────────────────
#  Part B: prefill + decode 混合并发
# ──────────────────────────────────────────────────────────────────────

def bench_mixed(tok, *, bs_decode: int, bs_prefill: int, plen: int,
                decode_tokens: int, insert_step: int, mode: str,
                mix: bool = True, reps: int = 3) -> dict:
    """波 1 进 decode，insert_step 步后追加波 2 prefill。返回端到端完成时间。"""
    cfg = {
        "eager": dict(enforce_eager=True, prefill_cuda_graph=False),
        "cg":    dict(enforce_eager=False, prefill_cuda_graph=True),
    }[mode]
    eng = make_engine(mix_prefill_decode=mix, **cfg)
    p1 = [build_prompt(tok, plen) for _ in range(bs_decode)]
    p2 = [build_prompt(tok, plen) for _ in range(bs_prefill)]

    # 预热
    seqs0 = [eng.add_request(p, SamplingParams(max_tokens=1, temperature=0.0)) for p in p1]
    while not eng.scheduler.is_finished():
        eng.step()
    teardown(eng)
    eng = None

    eng = make_engine(mix_prefill_decode=mix, **cfg)
    walls: list[float] = []
    for _ in range(reps):
        seqs1 = [eng.add_request(p, SamplingParams(max_tokens=decode_tokens, temperature=0.0)) for p in p1]
        t0 = time.perf_counter()
        step = 0
        added2 = False
        while not eng.scheduler.is_finished():
            if not added2 and step >= insert_step:
                for p in p2:
                    eng.add_request(p, SamplingParams(max_tokens=1, temperature=0.0))
                added2 = True
            eng.step()
            step += 1
        walls.append((time.perf_counter() - t0) * 1000)
    teardown(eng)

    return {
        "mode": mode, "bs_decode": bs_decode, "bs_prefill": bs_prefill,
        "plen": plen, "decode_tokens": decode_tokens, "insert_step": insert_step,
        "mix": mix, "wall_mean_ms": round(statistics.mean(walls), 3),
        "wall_min_ms": round(min(walls), 3),
    }


# ──────────────────────────────────────────────────────────────────────
#  main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--part", default="A", choices=["A", "B", "both"])
    p.add_argument("--out", default="/tmp/bench_concurrent.json")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    results: dict = {}

    if args.part in ("A", "both"):
        print("=== Part A: 纯 prefill 并发 ===")
        rows = []
        for bs in (1, 2, 4, 8):
            for plen in (129, 513):
                for mode in ("eager", "cg"):
                    r = bench_pure_prefill(tok, bs=bs, plen=plen, mode=mode)
                    rows.append(r)
                    flag = " [CG]" if r["cg_used"] else " [eager fallback]"
                    print(f"  bs={bs} len={plen:4d} {mode:7s}: wall={r['wall_mean_ms']:8.2f}ms"
                          f"  ttft={r['ttft_mean_ms']:8.2f}ms  steps={r['steps']}{flag}")
        results["partA"] = rows

    if args.part in ("B", "both"):
        print("\n=== Part B: prefill+decode 混合并发 ===")
        rows = []
        scenarios = [
            dict(bs_decode=4, bs_prefill=4, plen=129, decode_tokens=32, insert_step=3),
            dict(bs_decode=8, bs_prefill=4, plen=129, decode_tokens=32, insert_step=3),
            dict(bs_decode=8, bs_prefill=8, plen=129, decode_tokens=32, insert_step=5),
        ]
        for sc in scenarios:
            for mix in (True, False):
                for mode in ("eager", "cg"):
                    r = bench_mixed(tok, mode=mode, mix=mix, **sc)
                    rows.append(r)
                    print(f"  d={sc['bs_decode']} p={sc['bs_prefill']} mix={mix} {mode:7s}: "
                          f"e2e={r['wall_mean_ms']:8.2f}ms")
        results["partB"] = rows

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
