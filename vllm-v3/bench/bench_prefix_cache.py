"""前缀命中（Prefix Cache）实测：无缓存 vs hash vs radix。

v2/v3 的前缀缓存收益对比。同进程内 v2/v3 的 `src` 包名冲突，
因此用 `--engine v2|v3` 指定单版本运行（独立进程天然隔离）。

测试流程（每个 backend 一循环）：
1. seed 请求（长 prompt + max_tokens=1）把 KV 发布进索引
2. 完全命中：prompt 与 seed 完全相同 → 应跳过 prefill
3. 部分命中：前 3/4 相同 + 后 1/4 不同 → 只算续 chunk
指标：TTFT（add_request → 第一个 token）、命中 token 数。

v3 block_size=256（命中粒度 256 token）；v2 block_size=16。

用法:
    python bench_prefix_cache.py --engine v3 --backend all
    python bench_prefix_cache.py --engine v2 --backend hash
    python bench_prefix_cache.py --engine v3 --prompt-len 1024 --partial 768
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL = str(Path.home() / "huggingface" / "Qwen3-0.6B")


def make_engine(engine: str, backend: str):
    sys.path.insert(0, str(ROOT) if engine == "v3" else str(ROOT.parent / "vllm-v2"))
    if engine == "v3":
        from src import Engine, EngineConfig
        from src.sampling_params import SamplingParams

        cfg = EngineConfig(
            model=MODEL,
            enforce_eager=True,
            prefill_cuda_graph=False,
            dtype="auto",
            context_len=4096,
            max_num_seqs=8,
            max_num_batched_tokens=8192,
            mix_prefill_decode=True,
            kvcache_block_size=256,
            prefix_backend=backend,
            gpu_memory_utilization=0.5,
        )
    else:
        from src.config import EngineConfig
        from src.engine import Engine
        from src.sampling_params import SamplingParams

        cfg = EngineConfig(
            model=MODEL,
            context_len=4096,
            prefix_backend=backend,
            block_size=16,
            max_num_seqs=8,
            max_num_batched_tokens=8192,
            mix_prefill_decode=True,
        )
    return Engine(cfg), SamplingParams


def run_request(engine, sp_cls, token_ids: list[int], max_tokens: int = 1) -> float:
    """提交请求并跑到首个 completion token，返回 wall 时间（ms）。"""
    engine.add_request(token_ids, sp_cls(temperature=0.0, max_tokens=max_tokens))
    t0 = time.perf_counter()
    while not engine.scheduler.is_finished():
        outputs, _ = engine.step()
        if outputs:
            return (time.perf_counter() - t0) * 1000
    return (time.perf_counter() - t0) * 1000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=("v2", "v3"), required=True)
    p.add_argument("--backend", default="all", help="none/hash/radix/all")
    p.add_argument("--prompt-len", type=int, default=1024)
    p.add_argument("--partial", type=int, default=768, help="部分命中前缀相同 token 数")
    args = p.parse_args()

    backends = ["none", "hash", "radix"] if args.backend == "all" else [args.backend]
    print(f"# 前缀命中实测 engine={args.engine} "
          f"(prompt_len={args.prompt_len}, partial_hit={args.partial})\n")

    rows = []
    for backend in backends:
        engine, sp_cls = make_engine(args.engine, backend)
        print(f"[info] engine={args.engine} backend={backend} ready")

        # warmup：触发首次 flashinfer/triton 编译，不计入统计
        run_request(engine, sp_cls, [7, 13, 29, 31, 47], max_tokens=1)
        engine.reset_kv()

        seed = [100 + i % 997 for i in range(args.prompt_len)]
        t_seed = run_request(engine, sp_cls, seed, max_tokens=1)
        seed_cached = getattr(engine, "last_num_cached_tokens", 0)
        print(f"[seed] {t_seed:.2f} ms (cached={seed_cached})")

        hit = [100 + i % 997 for i in range(args.prompt_len)]
        t_full = run_request(engine, sp_cls, hit, max_tokens=1)
        full_cached = getattr(engine, "last_num_cached_tokens", 0)
        print(f"[full-hit] {t_full:.2f} ms (cached={full_cached})")

        partial_tok = [100 + i % 997 for i in range(args.prompt_len)]
        for i in range(args.partial, args.prompt_len):
            partial_tok[i] = (partial_tok[i] + 31) % 1000
        t_partial = run_request(engine, sp_cls, partial_tok, max_tokens=1)
        partial_cached = getattr(engine, "last_num_cached_tokens", 0)
        print(f"[partial-hit] {t_partial:.2f} ms (cached={partial_cached})")

        rows.append((backend, t_seed, t_full, t_partial, full_cached, partial_cached))
        engine.destroy() if hasattr(engine, "destroy") else None
        del engine
        gc.collect()
        torch.cuda.empty_cache()

    print("\n| backend | seed (ms) | 完全命中 (ms) | 部分命中 (ms) | 全命中加速 | 部分命中加速 | 全命中 cached | 部分命中 cached |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for backend, t_seed, t_full, t_partial, fc, pc in rows:
        print(f"| {backend} | {t_seed:.2f} | {t_full:.2f} | {t_partial:.2f} | "
              f"{t_seed / t_full:.2f}x | {t_seed / t_partial:.2f}x | {fc} | {pc} |")

    out = Path(__file__).resolve().parent / "out"
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# 前缀命中实测 engine={args.engine} "
             f"(prompt_len={args.prompt_len}, partial_hit={args.partial})", ""]
    lines += ["| backend | seed (ms) | 完全命中 (ms) | 部分命中 (ms) | 全命中加速 | 部分命中加速 | 全命中 cached | 部分命中 cached |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for backend, t_seed, t_full, t_partial, fc, pc in rows:
        lines.append(
            f"| {backend} | {t_seed:.2f} | {t_full:.2f} | {t_partial:.2f} | "
            f"{t_seed / t_full:.2f}x | {t_seed / t_partial:.2f}x | {fc} | {pc} |"
        )
    fname = f"prefix_cache_{args.engine}.md"
    (out / fname).write_text("\n".join(lines) + "\n")
    print(f"\n→ saved to {out / fname}")


if __name__ == "__main__":
    main()
