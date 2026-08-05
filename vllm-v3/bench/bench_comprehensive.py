"""v3 综合性能测试：短prompt独立请求 vs 长文本多轮对话/高并发。

场景 A（piece CG 主场）:
  - 短 prompt（128/256/512）独立请求、无前缀缓存
  - 对比 eager vs cg（decode CG + prefill piece CG）
  - 单请求 TTFT + 独立并发吞吐

场景 B（piece CG 失效区）:
  - 长文本 prefill（2048/4096/8192），看 CG 是否回退 eager
  - 多轮对话：连续 N 轮，每轮 prompt = 历史 + 新问题（前缀逐步增长）
  - 前缀缓存 on/off 对比（hash backend）
  - 高并发共享前缀：多条相同前缀的请求
"""
from __future__ import annotations

import gc
import os
import statistics
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "/home/xingyongwang/code/mycuda/dzyy-vllm/vllm-v3")

import torch

from src import Engine, EngineConfig, SamplingParams

MODEL = "/home/xingyongwang/huggingface/Qwen3-0.6B"


def sync():
    torch.cuda.synchronize()


def mean(xs):
    return statistics.mean(xs) if xs else 0.0


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else 0.0


def make_prompt_tokens(tokenizer, target_len):
    chunk = (
        "The quick brown fox jumps over the lazy dog. "
        "Explain continuous batching for LLM inference. "
    ) * 4
    ids = []
    while len(ids) < target_len:
        ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
    return ids[:target_len]


def make_dialogue_tokens(tokenizer, n_turns, turn_len):
    """构造 n_turns 轮对话：每轮前缀 = 前一轮前缀 + 新 turn_len token。"""
    chunk = "What is the capital of France? The capital of France is Paris. "
    q = tokenizer.encode(chunk, add_special_tokens=False)
    base = make_prompt_tokens(tokenizer, 64)
    turns = []
    ctx = list(base)
    for _ in range(n_turns):
        while len(q) < turn_len:
            q.extend(tokenizer.encode(" ", add_special_tokens=False))
            q.extend(make_prompt_tokens(tokenizer, 16))
        ctx = ctx + q[:turn_len]
        turns.append(list(ctx))
    return turns


def build_engine(mode: str, prefix: str = "none", kv_blocks: int = 384):
    if mode == "cg":
        cfg = dict(enforce_eager=False, prefill_cuda_graph=True)
    elif mode == "eager":
        cfg = dict(enforce_eager=True, prefill_cuda_graph=False)
    else:
        raise ValueError(mode)
    eng = Engine(EngineConfig(
        model=MODEL,
        context_len=16384,
        dtype="auto",
        prefix_backend=prefix,
        max_num_seqs=64,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.6,
        num_kvcache_blocks=kv_blocks,
        **cfg,
    ))
    return eng


def measure_ttft(eng, prompt_ids, max_tokens=1):
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
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
    return first * 1000 if first is not None else (time.perf_counter() - t0) * 1000


def bench_ttft(eng, lens, iters=5):
    out = {}
    for n in lens:
        tokens = make_prompt_tokens(eng.tokenizer, n)
        xs = [measure_ttft(eng, tokens) for _ in range(iters)]
        out[n] = {"med": round(median(xs), 2), "min": round(min(xs), 2)}
    return out


def bench_concurrent(eng, n_reqs, prompt_len, n_out=16):
    sp = SamplingParams(max_tokens=n_out, temperature=0.0)
    seqs = [eng.add_request(make_prompt_tokens(eng.tokenizer, prompt_len), sp)
            for _ in range(n_reqs)]
    sync()
    t0 = time.perf_counter()
    while not eng.scheduler.is_finished():
        eng.step()
    sync()
    dt = time.perf_counter() - t0
    total = sum(s.num_completion_tokens for s in seqs)
    return {"n": n_reqs, "e2e_ms": round(dt * 1000, 1), "tok_s": round(total / dt, 1),
            "prefill_tokens": n_reqs * prompt_len}


def bench_dialogue(eng, n_turns=6, turn_len=128, max_new=8):
    """多轮对话：每轮 = 完整上下文 + 新问题。前缀缓存 on/off 对比。"""
    turns = make_dialogue_tokens(eng.tokenizer, n_turns, turn_len)
    turn_ttfts = []
    for i, ctx in enumerate(turns):
        ms = measure_ttft(eng, ctx, max_tokens=max_new)
        turn_ttfts.append(ms)
        prefix_hit = eng.kv.last_num_cached_tokens
        print(f"    turn={i+1} ctx_len={len(ctx):>5d} TTFT={ms:7.2f}ms  prefix_hit={prefix_hit}")
    return {"turns": turn_ttfts}


def bench_shared_prefix(eng, n_reqs, prefix_len, suffix_len=64, n_out=8):
    """高并发共享前缀：n_reqs 条请求共享相同前缀。"""
    sp = SamplingParams(max_tokens=n_out, temperature=0.0)
    prefix = make_prompt_tokens(eng.tokenizer, prefix_len)
    seqs = []
    for _ in range(n_reqs):
        prompt = prefix + make_prompt_tokens(eng.tokenizer, suffix_len)
        seqs.append(eng.add_request(prompt, sp))
    sync()
    t0 = time.perf_counter()
    while not eng.scheduler.is_finished():
        eng.step()
    sync()
    dt = time.perf_counter() - t0
    total = sum(s.num_completion_tokens for s in seqs)
    return {"n": n_reqs, "prefix": prefix_len, "e2e_ms": round(dt * 1000, 1),
            "tok_s": round(total / dt, 1)}


def main():
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # ---------- 场景 A：短 prompt 独立请求，无前缀缓存 ----------
    print("\n[场景 A] 短 prompt 独立请求（无前缀缓存）: eager vs cg")
    lens = [128, 256, 512]
    eng_e = build_engine("eager")
    r_e = bench_ttft(eng_e, lens)
    eng_e.destroy(); del eng_e; gc.collect(); torch.cuda.empty_cache()
    eng_c = build_engine("cg")
    r_c = bench_ttft(eng_c, lens)
    eng_c.destroy(); del eng_c; gc.collect(); torch.cuda.empty_cache()
    print("  TTFT (ms):")
    print(f"    {'len':>6s} {'eager':>8s} {'cg':>8s} {'CG加速':>8s}")
    for n in lens:
        print(f"    {n:>6d} {r_e[n]['med']:>8.2f} {r_c[n]['med']:>8.2f} {r_e[n]['med']/r_c[n]['med']:>6.1f}x")

    print("\n  [场景A] 独立并发 (bs=4, len=128):")
    eng_e = build_engine("eager")
    ce = bench_concurrent(eng_e, 4, 128)
    eng_e.destroy(); del eng_e; gc.collect(); torch.cuda.empty_cache()
    eng_c = build_engine("cg")
    cc = bench_concurrent(eng_c, 4, 128)
    eng_c.destroy(); del eng_c; gc.collect(); torch.cuda.empty_cache()
    print(f"    eager: e2e={ce['e2e_ms']}ms  {ce['tok_s']} tok/s")
    print(f"    cg:    e2e={cc['e2e_ms']}ms  {cc['tok_s']} tok/s  ({ce['e2e_ms']/cc['e2e_ms']:.1f}x)")

    # ---------- 场景 B：长文本 / 多轮对话 / 高并发 ----------
    print("\n[场景 B1] 长文本 prefill TTFT (cg 模式，超过网格后回退 eager):")
    eng_l = build_engine("cg")
    lens_b = [1024, 2048, 4096, 8192]
    r_b = bench_ttft(eng_l, lens_b, iters=3)
    for n in lens_b:
        print(f"    len={n:>5d}: TTFT(med)={r_b[n]['med']:8.2f}ms")
    eng_l.destroy()
    del eng_l
    gc.collect()
    torch.cuda.empty_cache()

    print("\n[场景 B2] 多轮对话（前缀增长，prefix_cache OFF vs ON）:")
    print("  -- prefix_backend=none --")
    eng_d0 = build_engine("cg", prefix="none")
    d0 = bench_dialogue(eng_d0, n_turns=5, turn_len=128)
    eng_d0.destroy()
    del eng_d0
    gc.collect()
    torch.cuda.empty_cache()
    print("  -- prefix_backend=hash --")
    eng_d1 = build_engine("cg", prefix="hash")
    d1 = bench_dialogue(eng_d1, n_turns=5, turn_len=128)
    eng_d1.destroy()
    del eng_d1
    gc.collect()
    torch.cuda.empty_cache()

    print("\n[场景 B3] 高并发共享前缀 (bs=16, prefix=512):")
    eng_s0 = build_engine("eager", prefix="none")
    s0 = bench_shared_prefix(eng_s0, 16, 512)
    eng_s0.destroy(); del eng_s0; gc.collect(); torch.cuda.empty_cache()
    eng_s1 = build_engine("eager", prefix="hash")
    s1 = bench_shared_prefix(eng_s1, 16, 512)
    eng_s1.destroy(); del eng_s1; gc.collect(); torch.cuda.empty_cache()
    eng_s2 = build_engine("cg", prefix="none")
    s2 = bench_shared_prefix(eng_s2, 16, 512)
    eng_s2.destroy(); del eng_s2; gc.collect(); torch.cuda.empty_cache()
    print(f"    eager+no-prefix: e2e={s0['e2e_ms']}ms  {s0['tok_s']} tok/s")
    print(f"    eager+hash-prefix: e2e={s1['e2e_ms']}ms  {s1['tok_s']} tok/s")
    print(f"    cg+no-prefix: e2e={s2['e2e_ms']}ms  {s2['tok_s']} tok/s")


if __name__ == "__main__":
    main()
