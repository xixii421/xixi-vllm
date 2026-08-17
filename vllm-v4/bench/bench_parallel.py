"""v4 并行性能对比：单卡 vs TP=2 vs PP=2（greedy decode，enforce_eager）。

统一 enforce_eager=True / torch_compile=False，隔离 TP/PP 通信开销：
- 单请求 TTFT（prompt=256）
- 单请求 decode tok/s
- batch=8 聚合吞吐 tok/s

用法（在 vllm-v4/ 下）:
    ../.venv2/bin/python bench/bench_parallel.py --tp 1 --pp 1 --tag "tp1pp1"
    ../.venv2/bin/python bench/bench_parallel.py --tp 2 --pp 1 --tag "tp2pp1"
    ../.venv2/bin/python bench/bench_parallel.py --tp 1 --pp 2 --tag "tp1pp2"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import Engine, EngineConfig, SamplingParams  # noqa: E402

MODEL = str(Path.home() / "huggingface" / "Qwen3-0.6B")
OUT = Path(__file__).resolve().parent / "out"
PROMPT_LEN = 256
MAX_TOKENS = 128


def dummy_ids(n: int, salt: int = 0) -> list[int]:
    n = max(1, n)
    return [((i * 7919 + 10007 + salt * 97) % 150000) + 100 for i in range(n)]


def make_engine(tp: int, pp: int, model: str = MODEL) -> Engine:
    return Engine(
        EngineConfig(
            model=model,
            tp_size=tp,
            pp_size=pp,
            enforce_eager=True,
            torch_compile=False,
            dtype="bfloat16",
            context_len=4096,
            max_num_seqs=16,
            max_num_batched_tokens=4096,
            prefix_backend="none",
            gpu_memory_utilization=0.6,
            num_kvcache_blocks=64,
        )
    )


def run(tp: int, pp: int, tag: str, model: str = MODEL) -> None:
    engine = make_engine(tp, pp, model)
    rows: list[str] = []

    # 预热
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

    # batch=8 聚合吞吐
    b = 8
    sps = [
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=MAX_TOKENS)
        for _ in range(b)
    ]
    seqs = [
        engine.add_request(dummy_ids(PROMPT_LEN, salt=1000 + i), sps[i])
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

    header = "config | batch | TTFT (ms) | decode tok/s"
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "bench_parallel.txt"
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
    p.add_argument("--tp", type=int, required=True)
    p.add_argument("--pp", type=int, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--model", default=MODEL)
    args = p.parse_args()
    run(args.tp, args.pp, args.tag, str(Path(args.model).expanduser()))


if __name__ == "__main__":
    main()
