"""v4 vs ullm 分布式性能对比（同配置下分别测 TP / PP）。

统一模型 / prompt 长度 / 生成长度 / batch，对比：
- 单请求 TTFT (ms)
- 单请求 decode tok/s
- batch 聚合 decode tok/s

两个后端在相同 GPU 上跑（tp=2 / pp=2 均用 cuda:0 + cuda:1）：
    # TP
    ../.venv2/bin/python bench/bench_vs_ullm.py --backend v4   --tp 2 --pp 1
    ../.venv2/bin/python bench/bench_vs_ullm.py --backend ullm --tp 2 --pp 1
    # PP
    ../.venv2/bin/python bench/bench_vs_ullm.py --backend v4   --tp 1 --pp 2
    ../.venv2/bin/python bench/bench_vs_ullm.py --backend ullm --tp 1 --pp 2
    # v4 可额外用 --cg / --compile 开图与编译
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ULLM_ROOT = ROOT.parent / "ullm"
if str(ULLM_ROOT) not in sys.path:
    sys.path.insert(0, str(ULLM_ROOT))

from src import Engine, EngineConfig, SamplingParams  # noqa: E402

MODEL = str(Path.home() / "huggingface" / "Qwen3-0.6B")
OUT = Path(__file__).resolve().parent / "out"
PROMPT_LEN = 256
MAX_TOKENS = 128
BATCH = 8


def dummy_ids(n: int, salt: int = 0) -> list[int]:
    n = max(1, n)
    return [((i * 7919 + 10007 + salt * 97) % 150000) + 100 for i in range(n)]


# ------------------------------------------------------------------ #
# v4（sync）
# ------------------------------------------------------------------ #

def bench_v4(
    tp: int, pp: int, model: str, *, cg: bool, compile_: bool
) -> tuple[float, float, float]:
    engine = Engine(
        EngineConfig(
            model=model,
            tp_size=tp,
            pp_size=pp,
            enforce_eager=not cg,
            torch_compile=compile_,
            dtype="bfloat16",
            context_len=4096,
            max_num_seqs=max(BATCH, 16),
            max_num_batched_tokens=4096,
            prefix_backend="none",
            gpu_memory_utilization=0.6,
            num_kvcache_blocks=64,
        )
    )
    try:
        # 预热
        for it in range(3):
            seq = engine.add_request(
                dummy_ids(PROMPT_LEN, salt=it),
                SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True),
            )
            while not seq.is_finished:
                engine.step()
        torch.cuda.synchronize()

        # 单请求 TTFT + decode
        seq = engine.add_request(
            dummy_ids(PROMPT_LEN, salt=100),
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=MAX_TOKENS),
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ttft = None
        while not seq.is_finished:
            engine.step()
            if ttft is None and seq.num_completion_tokens > 0:
                torch.cuda.synchronize()
                ttft = time.perf_counter() - t0
        torch.cuda.synchronize()
        e2e = time.perf_counter() - t0
        n_out = seq.num_completion_tokens
        decode_tps = (n_out - 1) / (e2e - ttft) if e2e > ttft else 0.0

        # batch 聚合吞吐
        sps = [
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=MAX_TOKENS)
            for _ in range(BATCH)
        ]
        seqs = [
            engine.add_request(dummy_ids(PROMPT_LEN, salt=1000 + i), sps[i])
            for i in range(BATCH)
        ]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while not engine.scheduler.is_finished():
            engine.step()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        n = sum(s.num_completion_tokens for s in seqs)
        batch_tps = n / wall
        return ttft * 1000, decode_tps, batch_tps
    finally:
        engine.destroy()


# ------------------------------------------------------------------ #
# ullm（async）
# ------------------------------------------------------------------ #

async def _consume(gen) -> None:
    async for _ in gen:
        pass


async def bench_ullm_async(tp: int, pp: int, model: str) -> tuple[float, float, float]:
    from ullm import LLM, EngineConfig as UConfig, SamplingParams as USamplingParams

    llm = LLM(
        UConfig(
            model=model,
            gpu_memory_utilization=0.6,
            max_bs=max(BATCH, 16),
            tp_size=tp,
            pp_size=pp,
            context_len=4096,
            device_ids=[0, 1] if tp * pp > 1 else [0],
        )
    )
    try:
        # 预热
        await _consume(
            llm.generate(
                dummy_ids(PROMPT_LEN),
                USamplingParams(temperature=0.0, ignore_eos=True, max_new_tokens=4),
            )
        )

        # 单请求 TTFT + decode
        t0 = time.perf_counter()
        first = None
        async for _ in llm.generate(
            dummy_ids(PROMPT_LEN, salt=100),
            USamplingParams(
                temperature=0.0, ignore_eos=True, max_new_tokens=MAX_TOKENS
            ),
        ):
            if first is None:
                first = time.perf_counter() - t0
        e2e = time.perf_counter() - t0
        ttft = first
        decode_tps = (MAX_TOKENS - 1) / (e2e - first) if e2e > first else 0.0

        # batch 聚合吞吐
        prompts = [dummy_ids(PROMPT_LEN, salt=1000 + i) for i in range(BATCH)]
        sps = [
            USamplingParams(
                temperature=0.0, ignore_eos=True, max_new_tokens=MAX_TOKENS
            )
            for _ in range(BATCH)
        ]
        t0 = time.perf_counter()
        await asyncio.gather(
            *[_consume(llm.generate(p, s)) for p, s in zip(prompts, sps)]
        )
        wall = time.perf_counter() - t0
        batch_tps = BATCH * MAX_TOKENS / wall
        return ttft * 1000, decode_tps, batch_tps
    finally:
        await llm.shutdown()


def bench_ullm(tp: int, pp: int, model: str) -> tuple[float, float, float]:
    return asyncio.run(bench_ullm_async(tp, pp, model))


# ------------------------------------------------------------------ #

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["v4", "ullm"], required=True)
    p.add_argument("--tp", type=int, required=True)
    p.add_argument("--pp", type=int, required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--cg", action="store_true", help="v4 专用：开 decode CUDA Graph")
    p.add_argument("--compile", action="store_true", help="v4 专用：开 prefill torch.compile")
    args = p.parse_args()

    model = str(Path(args.model).expanduser())
    if args.backend == "v4":
        ttft, dec, batch = bench_v4(
            args.tp, args.pp, model, cg=args.cg, compile_=args.compile
        )
        tag = f"v4-tp{args.tp}pp{args.pp}" + ("-cg" if args.cg else "") + (
            "-cmp" if args.compile else ""
        )
    else:
        ttft, dec, batch = bench_ullm(args.tp, args.pp, model)
        tag = f"ullm-tp{args.tp}pp{args.pp}"

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "bench_vs_ullm.txt"
    header = "config | TTFT (ms) | decode tok/s (x1) | decode tok/s (batch)"
    first = not path.exists() or path.stat().st_size == 0
    line = f"{tag} | {ttft:.2f} | {dec:.1f} | {batch:.1f}"
    with path.open("a") as f:
        if first:
            f.write(header + "\n")
        f.write(line + "\n")
    print(line)
    print(f"→ {path}")


if __name__ == "__main__":
    main()
