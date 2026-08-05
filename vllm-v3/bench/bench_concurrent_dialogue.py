"""并发 × 连续问答：piece CG 命中率与开销分析。

场景：N 个并发用户，每人 R 轮对话。每轮所有用户同时提交一条消息
（同一 prompt 前缀 → hash 前缀缓存命中），引擎 step 直到全部完成。

三种配置对比：
  A. v3-full   hash + decodeCG + pieceCG  —— 完整 v3
  B. hash-eager hash + 全部 CG 关         —— 仅前缀缓存
  C. none-cg   none + decodeCG + pieceCG  —— piece CG 无前缀缓存

输出：
  · 每轮平均 TTFT、e2e
  · 每条 prefill/decode 路径命中次数（piece CG / eager / decode CG）
  · piece CG 的实际命中率与 padding 浪费
  · 显存增量与初始化时间（piece CG 开销）
"""
from __future__ import annotations

import argparse
import gc
import os
import random
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import EngineConfig
from src.engine.engine import Engine
from src.sampling_params import SamplingParams

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B")
KV_BLOCKS = 96
PROMPT0 = 300       # 首轮 prompt 长度
GEN = 16            # 每轮生成长度
TAIL = 220          # 每轮追加的新文本长度


def make_prompt(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(0, 5000) for _ in range(n)]


def build_engine(prefix: str, cg: bool, decode_cg: bool) -> tuple[Engine, float, dict]:
    """返回 (engine, init_s, mem)。decode_cg 用 enforce_eager=False 启用。"""
    cfg = EngineConfig(
        model=MODEL, context_len=4096, max_num_batched_tokens=8192,
        prefix_backend=prefix, num_kvcache_blocks=KV_BLOCKS,
        enforce_eager=not decode_cg,
        mix_prefill_decode=True,
        prefill_cuda_graph=cg,
    )
    t0 = time.perf_counter()
    eng = Engine(cfg)
    init_s = time.perf_counter() - t0
    torch.cuda.synchronize()
    mem = {
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "alloc_gb": torch.cuda.memory_allocated() / 1024**3,
    }
    return eng, init_s, mem


def attach_counters(eng: Engine) -> dict:
    """monkeypatch _run_batch，统计每条路径的实际执行次数。"""
    stats = {"piece_cg": 0, "eager_prefill": 0,
             "decode_cg": 0, "eager_decode": 0, "piece_pad_ratio": []}
    orig = eng.model_runner._run_batch

    def traced(seqs, *, is_prefill):
        if not is_prefill:
            if (eng.model_runner.cuda_graph_runner is not None
                    and eng.model_runner.cuda_graph_runner.can_use(len(seqs))):
                stats["decode_cg"] += 1
            else:
                stats["eager_decode"] += 1
            return orig(seqs, is_prefill=is_prefill)

        # prefill 路径判定（与 _run_batch 内部条件一致）
        if all(s.num_cached_tokens == 0 for s in seqs):
            lens = [s.num_scheduled_tokens for s in seqs]
            r = eng.model_runner.prefill_cg_runner
            if r is not None and r.can_use(len(seqs), lens):
                stats["piece_cg"] += 1
                piece = r.can_use(len(seqs), lens)
                stats["piece_pad_ratio"].append(piece[1] / max(lens))
            else:
                stats["eager_prefill"] += 1
        else:
            stats["eager_prefill"] += 1
        return orig(seqs, is_prefill=is_prefill)

    eng.model_runner._run_batch = traced
    return stats


def run_dialogue(eng: Engine, stats: dict, *, n_users: int, rounds: int) -> dict:
    """N 用户并发，每轮所有用户同时提交。返回每轮 TTFT 与 e2e。"""
    sp = SamplingParams(temperature=0.0, max_tokens=GEN, top_p=None, top_k=None)
    ctxs = [make_prompt(PROMPT0, seed=100 + u) for u in range(n_users)]
    per_round_ttft: list[float] = []
    e2e0 = time.monotonic()
    for r in range(rounds):
        seqs = [eng.add_request(ctxs[u], sp) for u in range(n_users)]
        first_at: dict[int, float] = {}
        t0 = time.monotonic()
        while not all(s.is_finished for s in seqs):
            eng.step()
            for s in seqs:
                if s.seq_id not in first_at and s.num_completion_tokens >= 1:
                    first_at[s.seq_id] = (time.monotonic() - t0) * 1000
        per_round_ttft.append(sum(first_at.values()) / len(first_at))
        # 下一轮：上一轮上下文 + 生成 + 新 tail（→ 前缀命中）
        if r < rounds - 1:
            for u, s in enumerate(seqs):
                ctxs[u] = (ctxs[u]
                           + list(s.token_ids[s.num_prompt_tokens:])
                           + make_prompt(TAIL, seed=200 + u + r * 10))
    e2e = (time.monotonic() - e2e0) * 1000
    return {"per_round_ttft": per_round_ttft, "e2e_ms": e2e}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=4, help="并发用户数")
    ap.add_argument("--rounds", type=int, default=5, help="对话轮数")
    ap.add_argument("--configs", type=str, default="v3-full,hash-eager,none-cg")
    args = ap.parse_args()

    torch.cuda.empty_cache()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"场景: {args.users} 并发用户 × {args.rounds} 轮对话, "
          f"首轮{PROMPT0}tok/生成{GEN}/tail{TAIL}")

    configs = {
        "v3-full":   dict(prefix="hash", cg=True,  decode_cg=True),
        "hash-eager": dict(prefix="hash", cg=False, decode_cg=False),
        "none-cg":   dict(prefix="none", cg=True,  decode_cg=True),
    }
    results = {}
    for name in args.configs.split(","):
        name = name.strip()
        if name not in configs:
            continue
        kw = configs[name]
        eng, init_s, mem = build_engine(**kw)
        stats = attach_counters(eng)
        res = run_dialogue(eng, stats, n_users=args.users, rounds=args.rounds)
        results[name] = {
            "init_s": init_s, **mem,
            **res,
            "paths": {k: v for k, v in stats.items() if k != "piece_pad_ratio"},
            "piece_hits": stats["piece_cg"],
            "eager_prefills": stats["eager_prefill"],
            "piece_avg_pad": (sum(stats["piece_pad_ratio"]) / len(stats["piece_pad_ratio"])
                              if stats["piece_pad_ratio"] else 0),
        }
        print(f"\n[{name}] init={init_s:.1f}s mem={mem['reserved_gb']:.2f}GiB "
              f"e2e={res['e2e_ms']:.0f}ms")
        print(f"  rounds_ttft_ms={[f'{v:.1f}' for v in res['per_round_ttft']]}")
        print(f"  paths: pieceCG={stats['piece_cg']} "
              f"eager_prefill={stats['eager_prefill']} "
              f"decodeCG={stats['decode_cg']} eager_decode={stats['eager_decode']}")
        if stats["piece_cg"]:
            print(f"  piece CG: {stats['piece_cg']} 次命中, 平均 pad={results[name]['piece_avg_pad']:.2f}x")
        eng.destroy()
        del eng
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== 汇总 ===")
    hdr = f"{'config':<12}{'init_s':>7}{'mem':>7}{'e2e':>7}{'TTFT_r1':>9}{'TTFT_rN':>9}{'piece':>6}{'eager':>6}"
    print(hdr)
    for name, r in results.items():
        r1 = r["per_round_ttft"][0]
        rN = r["per_round_ttft"][-1]
        print(f"{name:<12}{r['init_s']:>7.1f}{r['reserved_gb']:>7.2f}{r['e2e_ms']:>7.0f}"
              f"{r1:>9.1f}{rN:>9.1f}"
              f"{r['piece_hits']:>6}{r['eager_prefills']:>6}")


if __name__ == "__main__":
    main()
