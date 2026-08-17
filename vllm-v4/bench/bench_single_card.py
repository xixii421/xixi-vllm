"""单卡 decode 吞吐对比：v3 vs v4（tp=1/pp=1，GPU 4070）。

greedy decode：单请求 TTFT + decode tok/s；batch=2/4/8 总吞吐。
v3 与 v4 的 ``src`` 包名冲突，按 ``--root`` 分进程跑。

用法（在 vllm-v4/ 下）:
    ../.venv2/bin/python bench/bench_single_card.py --root ../vllm-v3 --tag v3
    ../.venv2/bin/python bench/bench_single_card.py --root ../vllm-v4 --tag v4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

BENCH = Path(__file__).resolve().parent
OUT = BENCH / "out"
MODEL = str(Path.home() / "huggingface" / "Qwen3-0.6B")
PROMPT_LEN = 256
MAX_TOKENS = 128


def dummy_ids(n: int, salt: int = 0) -> list[int]:
    n = max(1, n)
    return [((i * 7919 + 10007 + salt * 97) % 150000) + 100 for i in range(n)]


def make_engine(Engine, EngineConfig):
    return Engine(
        EngineConfig(
            model=MODEL,
            enforce_eager=False,  # decode CUDA Graph
            torch_compile=False,
            dtype="auto",
            context_len=4096,
            max_num_seqs=8,
            max_num_batched_tokens=4096,
            mix_prefill_decode=True,
            kvcache_block_size=256,
            prefix_backend="none",
            gpu_memory_utilization=0.85,
        )
    )


def run(root: str, tag: str) -> None:
    root_path = Path(root).resolve()
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))
    from src import Engine, EngineConfig, SamplingParams  # noqa: E402

    engine = make_engine(Engine, EngineConfig)
    rows: list[str] = []

    sp_w = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
    for it in range(3):
        seq = engine.add_request(dummy_ids(PROMPT_LEN, salt=it), sp_w)
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
    rows.append(f"{tag} | 1 | {ttft * 1000:.2f} | {decode_tps:.1f}")

    # 多并发吞吐
    for b in (2, 4, 8):
        sps = [
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=MAX_TOKENS)
            for _ in range(b)
        ]
        seqs = [
            engine.add_request(dummy_ids(PROMPT_LEN, salt=1000 + b * 10 + i), sps[i])
            for i in range(b)
        ]
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
    path = OUT / "bench_single_card.txt"
    first = not path.exists() or path.stat().st_size == 0
    with path.open("a") as f:
        if first:
            f.write(header + "\n")
        for r in rows:
            f.write(r + "\n")
            print(r)
    print(f"→ {path}")
    engine.destroy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    run(args.root, args.tag)


if __name__ == "__main__":
    main()
