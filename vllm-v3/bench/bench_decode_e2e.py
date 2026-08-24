"""greedy decode 吞吐：v2 vs v3（单请求 + 多并发）。

greedy（temperature=0）为生产默认，直接体现 v3 对 v2 的核心加速
（decode CUDA Graph + FlashAttention batch decode）。v2 与 v3 的 src 包名
冲突，按 --engine 分进程跑。

单请求：prompt=256，max_tokens=128，测 TTFT 与 decode tok/s。
多并发：batch=1/2/4/8，prompt=256，max_tokens=128，测总吞吐。

用法（在 vllm-v3/ 下）:
    ../.venv2/bin/python bench/bench_decode_e2e.py --engine v2
    ../.venv2/bin/python bench/bench_decode_e2e.py --engine v3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

BENCH = Path(__file__).resolve().parent
V3_ROOT = BENCH.parent
REPO = V3_ROOT.parent
V2_ROOT = REPO / "vllm-v2"
OUT = BENCH / "out"
MODEL = str(Path.home() / "huggingface" / "Qwen3-0.6B")
PROMPT_LEN = 256
MAX_TOKENS = 128


def dummy_ids(n: int, salt: int = 0) -> list[int]:
    n = max(1, n)
    return [((i * 7919 + 10007 + salt * 97) % 150000) + 100 for i in range(n)]


def run_v2() -> None:
    if str(V2_ROOT) not in sys.path:
        sys.path.insert(0, str(V2_ROOT))
    from src.config import EngineConfig  # noqa: E402
    from src.engine import Engine  # noqa: E402
    from src.sampling_params import SamplingParams  # noqa: E402

    engine = Engine(
        EngineConfig(
            model=MODEL, dtype="auto", context_len=4096, block_size=16,
            prefix_backend="none", max_num_seqs=8, max_num_batched_tokens=4096,
            mix_prefill_decode=True,
        )
    )
    _run(engine, SamplingParams, "v2", destroy=None)


def run_v3() -> None:
    if str(BENCH) not in sys.path:
        sys.path.insert(0, str(BENCH))
    if str(V3_ROOT) not in sys.path:
        sys.path.insert(0, str(V3_ROOT))
    from common import destroy, make_engine  # noqa: E402
    from src import SamplingParams  # noqa: E402

    engine = make_engine(cg=True, compile=False, prefix="none", max_num_seqs=8, gpu_mem=0.85)
    _run(engine, SamplingParams, "v3", destroy=destroy)


def _run(engine, SP, tag: str, destroy) -> None:
    sp_w = SP(temperature=0.0, max_tokens=4, ignore_eos=True)
    for it in range(3):
        seq = engine.add_request(dummy_ids(PROMPT_LEN, salt=it), sp_w)
        while not seq.is_finished:
            engine.step()
    torch.cuda.synchronize()

    rows: list[str] = []

    # 单请求 TTFT + decode
    seq = engine.add_request(dummy_ids(PROMPT_LEN, salt=100), SP(temperature=0.0, ignore_eos=True, max_tokens=MAX_TOKENS))
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
    rows.append(f"{tag} | 1 | {ttft * 1000:.2f} | {decode_tps:.1f}")

    # 多并发吞吐
    for b in (2, 4, 8):
        sps = [SP(temperature=0.0, ignore_eos=True, max_tokens=MAX_TOKENS) for _ in range(b)]
        seqs = [engine.add_request(dummy_ids(PROMPT_LEN, salt=1000 + b * 10 + i), sps[i]) for i in range(b)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while not engine.scheduler.is_finished():
            engine.step()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        n = sum(s.num_completion_tokens for s in seqs)
        rows.append(f"{tag} | {b} | - | {n / wall:.1f}")

    header = "engine | batch | TTFT (ms) | decode tok/s"
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "bench_decode_e2e.txt"
    first = not path.exists() or path.stat().st_size == 0
    with path.open("a") as f:
        if first:
            f.write(header + "\n")
        for r in rows:
            f.write(r + "\n")
            print(r)
    print(f"→ {path}")
    if destroy is not None:
        destroy(engine)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=("v2", "v3"), required=True)
    args = p.parse_args()
    if args.engine == "v2":
        run_v2()
    else:
        run_v3()


if __name__ == "__main__":
    main()
