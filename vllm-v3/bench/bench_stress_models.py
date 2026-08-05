#!/usr/bin/env python3
"""v3 连续负载全栈压测：不同模型 × radix + prefill/decode CG。

在 4090 上完整测：
  · init 开销分解：权重加载 → decode CG 捕获 → piece CG 捕获 → KV
  · 连续对话负载：N 用户并发 R 轮，共享长前缀（radix 命中）
  · 指标：每轮 TTFT / e2e / 吞吐 / prefill 路径统计 / radix 命中 / 显存

用法:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=vllm-v3 python vllm-v3/bench/bench_stress_models.py
"""
from __future__ import annotations

import gc
import os
import random
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # 0 = 4090 D (24GiB)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import EngineConfig
from src.engine.engine import Engine
from src.sampling_params import SamplingParams

MODELS = {
    "0.6B": "/home/xingyongwang/huggingface/Qwen3-0.6B",
    "1.7B": "/home/xingyongwang/huggingface/Qwen3-1.7B",
    "4B":   "/home/xingyongwang/huggingface/Qwen3-4B",
    "8B":   "/home/xingyongwang/huggingface/Qwen3-8B",
}
# 每模型：并发用户 / 首轮 prompt / 每轮生成 / 每轮 tail / KV 自动推导
LOAD = {
    "0.6B": dict(users=8, prompt0=512, gen=32, tail=256),
    "1.7B": dict(users=8, prompt0=512, gen=32, tail=256),
    "4B":   dict(users=8, prompt0=512, gen=32, tail=256),
    "8B":   dict(users=4, prompt0=384, gen=32, tail=256),
}
ROUNDS = 5


def make_prompt(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(0, 5000) for _ in range(n)]


def build_engine(model_path: str, *, full: bool) -> tuple[Engine, dict]:
    """构建引擎。full=True 开 radix + decode/piece CG；full=False 仅 eager。"""
    cfg = EngineConfig(
        model=model_path, context_len=4096, max_num_seqs=16,
        max_num_batched_tokens=8192,
        prefix_backend="radix",
        num_kvcache_blocks=0,          # 自动按显存推导
        enforce_eager=not full,
        mix_prefill_decode=True,
        prefill_cuda_graph=full,
    )

    # ── init 分解：插桩 ModelRunner 的各捕获阶段 ──
    from src.engine import model_runner as mr_mod
    times: dict[str, float] = {}

    def _t(name):
        def deco(fn):
            def wrap(*a, **kw):
                t0 = time.perf_counter()
                out = fn(*a, **kw)
                times[name] = round((time.perf_counter() - t0), 3)
                return out
            return wrap
        return deco

    capture_names = ("_capture_cg", "_capture_prefill_cg")
    orig_fns = {n: getattr(mr_mod.ModelRunner, n) for n in capture_names if hasattr(mr_mod.ModelRunner, n)}
    for n, fn in orig_fns.items():
        setattr(mr_mod.ModelRunner, n, _t(n)(fn))

    try:
        t0 = time.perf_counter()
        eng = Engine(cfg)
        torch.cuda.synchronize()
        total_init = time.perf_counter() - t0
    finally:
        for n, fn in orig_fns.items():
            setattr(mr_mod.ModelRunner, n, fn)

    nblocks = cfg.num_kvcache_blocks
    mem = {
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
        "alloc_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "kv_blocks": nblocks,
    }
    info = {
        "total_init_s": round(total_init, 2),
        "capture_times": times,
        "mem": mem,
    }
    return eng, info


def attach_counters(eng: Engine) -> dict:
    stats = {"piece_cg": 0, "eager_prefill": 0,
             "decode_cg": 0, "eager_decode": 0, "radix_hit_tokens": 0,
             "prefix_hits": 0, "first_cached_map": {}}
    orig = eng.model_runner._run_batch

    def traced(seqs, *, is_prefill):
        if not is_prefill:
            if (eng.model_runner.cuda_graph_runner is not None
                    and eng.model_runner.cuda_graph_runner.can_use(len(seqs))):
                stats["decode_cg"] += 1
            else:
                stats["eager_decode"] += 1
            return orig(seqs, is_prefill=is_prefill)

        # radix 前缀命中统计（prefill 开始时 num_cached_tokens == 命中块数*block_size）
        for s in seqs:
            if s.num_cached_tokens > 0:
                stats["prefix_hits"] += 1
                stats["radix_hit_tokens"] += s.num_cached_tokens
            stats["first_cached_map"].setdefault(s.seq_id, s.num_cached_tokens)

        if all(s.num_cached_tokens == 0 for s in seqs):
            lens = [s.num_scheduled_tokens for s in seqs]
            r = eng.model_runner.prefill_cg_runner
            if r is not None and r.can_use(len(seqs), lens):
                stats["piece_cg"] += 1
            else:
                stats["eager_prefill"] += 1
        else:
            stats["eager_prefill"] += 1
        return orig(seqs, is_prefill=is_prefill)

    eng.model_runner._run_batch = traced
    return stats


def run_dialogue(eng: Engine, stats: dict, *, users: int, rounds: int,
                 prompt0: int, gen: int, tail: int) -> dict:
    """N 用户并发连续对话。每轮共享长前缀 → radix 命中。"""
    sp = SamplingParams(temperature=0.0, max_tokens=gen, top_p=None, top_k=None)
    ctxs = [make_prompt(prompt0, seed=100 + u) for u in range(users)]
    per_round = []
    t_tot = time.monotonic()
    total_gen = 0
    for r in range(rounds):
        seqs = [eng.add_request(ctxs[u], sp) for u in range(users)]
        first_at: dict[int, float] = {}
        t0 = time.monotonic()
        while not all(s.is_finished for s in seqs):
            eng.step()
            for s in seqs:
                if s.seq_id not in first_at and s.num_completion_tokens >= 1:
                    first_at[s.seq_id] = (time.monotonic() - t0) * 1000
        dt = (time.monotonic() - t0) * 1000
        gen_toks = sum(s.num_completion_tokens for s in seqs)
        total_gen += gen_toks
        # 前缀命中 = 首次 prefill 时已缓存 token 数（块对齐）
        hit_toks = [stats["first_cached_map"].get(s.seq_id, 0) for s in seqs]
        per_round.append({
            "round": r + 1,
            "ttft_ms": round(sum(first_at.values()) / max(1, len(first_at)), 1),
            "e2e_ms": round(dt, 1),
            "tok_s": round(gen_toks / (dt / 1000), 1),
            "prefix_hit": round(sum(hit_toks) / max(1, len(hit_toks)), 1),
        })
        if r < rounds - 1:
            for u, s in enumerate(seqs):
                ctxs[u] = (ctxs[u]
                           + list(s.token_ids[s.num_prompt_tokens:])
                           + make_prompt(tail, seed=200 + u + r * 10))
    wall = (time.monotonic() - t_tot) * 1000
    return {
        "rounds": per_round,
        "total_gen": total_gen,
        "e2e_total_ms": round(wall, 1),
        "throughput_tok_s": round(total_gen / (wall / 1000), 1),
    }


def main() -> None:
    only = os.environ.get("STRESS_ONLY", "")
    print(f"GPU: {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory/1024**3:.0f} GiB)")

    all_results = {}
    for name, path in MODELS.items():
        if only and name not in only:
            continue
        L = LOAD[name]
        print(f"\n{'='*70}\n  {name}  {path}\n{'='*70}", flush=True)

        # ── init 分解：先 eager（无 CG）作基线，再 full ──
        torch.cuda.empty_cache()
        eng_e, info_e = build_engine(path, full=False)
        print(f"[{name}] eager init={info_e['total_init_s']}s "
              f"kv_blocks={info_e['mem']['kv_blocks']} "
              f"peak={info_e['mem']['peak_gb']}GiB", flush=True)
        eng_e.destroy(); del eng_e; gc.collect(); torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        eng, info = build_engine(path, full=True)
        ct = info["capture_times"]
        print(f"[{name}] full init={info['total_init_s']}s  "
              f"decodeCG={ct.get('_capture_cg', 0)}s  "
              f"pieceCG={ct.get('_capture_prefill_cg', 0)}s  "
              f"kv={info['mem']['kv_blocks']}blocks  peak={info['mem']['peak_gb']}GiB", flush=True)

        stats = attach_counters(eng)
        res = run_dialogue(eng, stats, users=L["users"], rounds=ROUNDS,
                           prompt0=L["prompt0"], gen=L["gen"], tail=L["tail"])

        all_results[name] = {
            "load": L,
            "init": info,
            "init_eager_s": info_e["total_init_s"],
            "cg_capture_s": ct,
            "mem": info["mem"],
            "dialogue": res,
            "paths": {k: v for k, v in stats.items()
                      if k not in ("first_cached_map",)},
        }

        print(f"[{name}] 连续对话: e2e={res['e2e_total_ms']}ms 吞吐={res['throughput_tok_s']}tok/s")
        for rd in res["rounds"]:
            print(f"    R{rd['round']}: TTFT={rd['ttft_ms']}ms e2e={rd['e2e_ms']}ms "
                  f"tok/s={rd['tok_s']} prefix_hit={rd['prefix_hit']}")
        print(f"    路径: pieceCG={stats['piece_cg']} "
              f"eager_prefill={stats['eager_prefill']} "
              f"decodeCG={stats['decode_cg']} eager_decode={stats['eager_decode']} "
              f"radix_hits={stats['prefix_hits']}")

        eng.destroy(); del eng; gc.collect(); torch.cuda.empty_cache()

    import json
    print("\n\n=== FULL JSON ===")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
