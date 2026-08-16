"""跨引擎连续批处理吞吐：v3 / nano-vllm / 官方 vLLM。

三个引擎不能混在同一进程（包名与依赖冲突），按 --engine 分开跑。
nano 禁止 greedy，所以三边统一 temperature=0.6 + ignore_eos。

v3 / nano 走仓库根目录 .venv2；官方 vLLM 走 .venv-vllm。

同一 workload：16 条，prompt 100~400，out 50~150，seed=0。
dummy token 公式与 bench_e2e / common.dummy_ids 对齐。

用法（在 vllm-v3/ 下）:
    ../.venv2/bin/python bench/bench_compare.py --engine v3
    ../.venv2/bin/python bench/bench_compare.py --engine nano
    ../.venv-vllm/bin/python bench/bench_compare.py --engine vllm
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch

BENCH = Path(__file__).resolve().parent
V3_ROOT = BENCH.parent
REPO = V3_ROOT.parent
NANO_ROOT = REPO / "nano-vllm"
OUT = BENCH / "out"
MODEL = str(Path.home() / "huggingface" / "Qwen3-0.6B")


def dummy_ids(n: int, salt: int = 0) -> list[int]:
    n = max(1, n)
    return [((i * 7919 + 10007 + salt * 97) % 150000) + 100 for i in range(n)]


def workload(n: int, *, min_in=100, max_in=400, min_out=50, max_out=150):
    rng = random.Random(0)
    prompts = [dummy_ids(rng.randint(min_in, max_in), salt=i) for i in range(n)]
    max_tokens = [rng.randint(min_out, max_out) for _ in range(n)]
    return prompts, max_tokens


_HEADER = "engine | tag | wall (s) | out tokens | tok/s"


def _append(engine: str, tag: str, wall: float, n_out: int, want: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "bench_compare.txt"
    first = not path.exists() or path.stat().st_size == 0
    with path.open("a") as f:
        if first:
            f.write(_HEADER + "\n")
        tps = n_out / wall if wall > 0 else 0.0
        f.write(f"{engine} | {tag} | {wall:.2f} | {n_out}/{want} | {tps:.1f}\n")
    print(
        f"[result] engine={engine} ({tag}) wall={wall:.2f}s "
        f"output_tokens={n_out}/{want} throughput={n_out / wall if wall > 0 else 0:.1f} tok/s"
    )
    print(f"→ {path}")


def run_v3(prompts, max_tokens, device: str) -> tuple[float, int, str]:
    if str(BENCH) not in sys.path:
        sys.path.insert(0, str(BENCH))
    if str(V3_ROOT) not in sys.path:
        sys.path.insert(0, str(V3_ROOT))
    from common import destroy, drain, make_engine, warmup  # noqa: E402
    from src import SamplingParams  # noqa: E402

    torch.cuda.set_device(device)
    engine = make_engine(
        cg=True,
        compile=True,
        prefix="none",
        max_num_seqs=max(16, len(prompts)),
        gpu_mem=0.85,
    )
    warmup(engine, 16, max_tokens=4)
    seqs = [
        engine.add_request(
            p, SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=mt)
        )
        for p, mt in zip(prompts, max_tokens)
    ]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    drain(engine)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    n_out = sum(s.num_completion_tokens for s in seqs)
    destroy(engine)
    return wall, n_out, "decodeCG+compile"


def run_nano(prompts, max_tokens) -> tuple[float, int, str]:
    if str(NANO_ROOT) not in sys.path:
        sys.path.insert(0, str(NANO_ROOT))
    from nanovllm import LLM, SamplingParams  # noqa: E402

    llm = LLM(
        MODEL,
        enforce_eager=False,
        max_model_len=4096,
        max_num_seqs=max(16, len(prompts)),
        gpu_memory_utilization=0.55,
    )
    llm.generate(
        [[7, 13, 29, 31]],
        SamplingParams(temperature=0.6, max_tokens=4, ignore_eos=True),
        use_tqdm=False,
    )
    sps = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=mt)
        for mt in max_tokens
    ]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sps, use_tqdm=False)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    n_out = sum(len(o["token_ids"]) for o in outs)
    return wall, n_out, "CUDA Graph"


def run_vllm(prompts, max_tokens) -> tuple[float, int, str]:
    from vllm import LLM, SamplingParams  # noqa: E402

    try:
        from vllm import TokensPrompt  # noqa: E402

        reqs = [TokensPrompt(prompt_token_ids=p) for p in prompts]
        warm = [TokensPrompt(prompt_token_ids=[7, 13, 29, 31])]
    except Exception:
        reqs = [dict(prompt_token_ids=p) for p in prompts]
        warm = [dict(prompt_token_ids=[7, 13, 29, 31])]

    llm = LLM(
        model=MODEL,
        enforce_eager=False,
        max_model_len=4096,
        gpu_memory_utilization=0.55,
        max_num_seqs=max(16, len(prompts)),
    )
    llm.generate(
        warm,
        SamplingParams(temperature=0.6, max_tokens=4, ignore_eos=True),
        use_tqdm=False,
    )
    sps = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=mt)
        for mt in max_tokens
    ]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(reqs, sps, use_tqdm=False)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    n_out = sum(len(o.outputs[0].token_ids) for o in outs)
    return wall, n_out, "vllm 0.26 default"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=("v3", "nano", "vllm"), required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-seqs", type=int, default=16)
    args = p.parse_args()

    prompts, max_tokens = workload(args.num_seqs)
    want = sum(max_tokens)
    print(
        f"# compare engine={args.engine}  n={args.num_seqs}  "
        f"target_out≈{want}  T=0.6\n"
    )

    if args.engine == "v3":
        wall, n_out, tag = run_v3(prompts, max_tokens, args.device)
    elif args.engine == "nano":
        wall, n_out, tag = run_nano(prompts, max_tokens)
    else:
        wall, n_out, tag = run_vllm(prompts, max_tokens)
    _append(args.engine, tag, wall, n_out, want)


if __name__ == "__main__":
    main()
