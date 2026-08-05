#!/usr/bin/env python3
"""多 seq 并发验证：eager-FA2(varlen) vs eager-FlashInfer vs CG-FA2。

覆盖两种并发形态：
  1. 纯 prefill 并发（max_tokens=1）：多请求同批 prefill，bs>1
  2. prefill+decode 混合（部分 max_tokens>1）：多步里 prefill/decode 交错
比较各模式产出的首个 token / 完整输出是否一致。
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

os.environ.setdefault("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from src import Engine, EngineConfig, SamplingParams


def build_prompt(tok: AutoTokenizer, n: int) -> list[int]:
    chunk = ("The quick brown fox jumps over the lazy dog. "
             "Explain continuous batching for LLM inference. ") * 4
    ids: list[int] = []
    while len(ids) < n:
        ids.extend(tok.encode(chunk, add_special_tokens=False))
    return ids[:n]


def run_concurrent(eng: Engine, prompts: list[list[int]], max_tokens: int) -> list[list[int]]:
    """并发添加所有请求，step 到全部结束，返回每个 seq 的完整输出 token。"""
    seqs = [eng.add_request(ids, SamplingParams(max_tokens=max_tokens, temperature=0.0))
            for ids in prompts]
    nsteps = 0
    while not eng.scheduler.is_finished():
        eng.step()
        nsteps += 1
    out = [seq.token_ids[seq.num_prompt_tokens:] for seq in seqs]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/xingyongwang/huggingface/Qwen3-0.6B")
    p.add_argument("--lens", type=int, nargs="+", default=[129, 256, 513, 777])
    p.add_argument("--max-tokens", type=int, default=8)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [build_prompt(tok, n) for n in args.lens]

    def make_engine(**kw):
        return Engine(EngineConfig(
            model=args.model, context_len=16384, dtype="auto", prefix_backend="none",
            max_num_seqs=8, num_kvcache_blocks=256, gpu_memory_utilization=0.85, **kw,
        ))

    results: dict[str, list[list[int]]] = {}

    modes = [
        ("eager_fa2", dict(enforce_eager=True, prefill_cuda_graph=False)),
        ("eager_fi", dict(enforce_eager=True, prefill_cuda_graph=False)),
        ("cg_fa2", dict(enforce_eager=False, prefill_cuda_graph=True)),
    ]
    for name, kw in modes:
        eng = make_engine(**kw)
        if name == "eager_fi":  # 关掉 FA2 → FlashInfer paged prefill
            orig = eng.model_runner.backend.build_metadata
            eng.model_runner.backend.build_metadata = lambda *a, _orig=orig, **k: _orig(
                *a, **{**k, "use_fa2": False}
            )
        results[name] = run_concurrent(eng, prompts, args.max_tokens)
        eng.destroy()
        del eng
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[done] {name}: lens={args.lens} max_tokens={args.max_tokens}")

    ref = results["eager_fa2"]
    print("\n== 并发结果对比（逐请求首个 token / 完整输出）==")
    all_ok = True
    for name in ("eager_fi", "cg_fa2"):
        for i, (a, b) in enumerate(zip(ref, results[name])):
            ok = a == b
            all_ok &= ok
            print(f"  {name} req{i}: {b[:3]}{'...' if len(b) > 3 else ''} "
                  f"vs eager_fa2 {a[:3]}{'...' if len(a) > 3 else ''} "
                  f"{'MATCH' if ok else 'MISMATCH'}")
    print("\n=== " + ("ALL MATCH" if all_ok else "MISMATCH DETECTED") + " ===")


if __name__ == "__main__":
    main()
