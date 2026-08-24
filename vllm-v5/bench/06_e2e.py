"""v5 端到端连续批处理吞吐（与 v3 bench/06_e2e.py 相同口径）。

负载：16 条变长请求（prompt 100~400，out 32~64），固定 seed，greedy + ignore_eos。
v5 无 torch.compile prefill，只测 eager / cg 两档（与 v3 对应档直接对比）。

用法:
    PYTHONPATH=. python bench/06_e2e.py
    PYTHONPATH=. python bench/06_e2e.py --mode cg
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import Engine, EngineConfig, SamplingParams  # noqa: E402

MODEL = str(Path.home() / "huggingface" / "Qwen3-0.6B")
OUT_DIR = Path(__file__).resolve().parent / "out"


def dummy_ids(n: int, salt: int = 0) -> list[int]:
    n = max(1, n)
    return [((i * 7919 + 10007 + salt * 97) % 150000) + 100 for i in range(n)]


def make_engine(*, cg: bool, max_num_seqs: int, gpu_mem: float = 0.85) -> Engine:
    return Engine(
        EngineConfig(
            model=MODEL,
            enforce_eager=not cg,
            dtype="auto",
            context_len=4096,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=8192,
            gpu_memory_utilization=gpu_mem,
            enable_radix=False,
        )
    )


def warmup(engine: Engine, n: int, *, iters: int = 3, max_tokens: int = 4) -> None:
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    torch.cuda.synchronize()
    for it in range(iters):
        engine.add_request(dummy_ids(n, salt=it), sp)
        while not engine.scheduler.is_finished():
            engine.step()
    torch.cuda.synchronize()


def drain(engine: Engine) -> None:
    while not engine.scheduler.is_finished():
        engine.step()


def workload(n: int, *, min_in=100, max_in=400, min_out=32, max_out=64):
    rng = random.Random(0)
    prompts = [dummy_ids(rng.randint(min_in, max_in), salt=i) for i in range(n)]
    max_tokens = [rng.randint(min_out, max_out) for _ in range(n)]
    return prompts, max_tokens


def run_mode(cg: bool, prompts, max_tokens) -> tuple[float, int, float]:
    engine = make_engine(cg=cg, max_num_seqs=max(16, len(prompts)))
    warmup(engine, 16, max_tokens=4)
    seqs = [
        engine.add_request(
            p, SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=mt)
        )
        for p, mt in zip(prompts, max_tokens)
    ]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    drain(engine)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    n_out = sum(s.num_completion_tokens for s in seqs)
    engine.destroy()
    return wall, n_out, n_out / wall if wall > 0 else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-seqs", type=int, default=16)
    p.add_argument("--mode", choices=("eager", "cg", "all"), default="all")
    args = p.parse_args()
    torch.cuda.set_device(args.device)

    prompts, max_tokens = workload(args.num_seqs)
    modes = [("eager", False), ("cg", True)] if args.mode == "all" else [
        (args.mode, args.mode == "cg")
    ]
    want = sum(max_tokens)

    lines = [
        f"# v5 e2e throughput  (n={args.num_seqs}, target_out≈{want})",
        "",
        "| mode | wall (s) | out tokens | tok/s |",
        "| --- | ---: | ---: | ---: |",
    ]
    for s in lines:
        print(s)
    print()
    for tag, cg in modes:
        wall, n_out, tps = run_mode(cg, prompts, max_tokens)
        row = f"| {tag} | {wall:.2f} | {n_out}/{want} | {tps:.1f} |"
        print(row)
        lines.append(row)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "06_e2e.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
