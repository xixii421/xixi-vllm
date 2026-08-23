"""v4 并行性能对比：单卡 vs TP=2 vs PP=2（greedy decode）。

指标：
- 单请求 TTFT（prompt=256）
- 单请求 decode tok/s
- batch=8 聚合吞吐 tok/s

开关：--cg（decode CUDA Graph）、--compile（prefill torch.compile）。
大模型支持：--model、--device-ids、--pp-layer-counts、--num-kvcache-blocks。

用法（在 vllm-v4/ 下）:
    ../.venv2/bin/python bench/bench_parallel.py --tp 1 --pp 1 --tag "tp1pp1"
    ../.venv2/bin/python bench/bench_parallel.py --model ~/shared/models/Qwen3-8B --tp 1 --pp 1 --device-ids 1 --cg --tag "8b-tp1-cg"
    ../.venv2/bin/python bench/bench_parallel.py --model ~/shared/models/Qwen3-14B --tp 1 --pp 2 --pp-layer-counts 12,28 --tag "14b-pp2"

结果同时写入 bench/out/bench_parallel.txt（人类可读）与 bench/out/bench.jsonl（结构化，供绘图）。
"""

from __future__ import annotations

import argparse
import json
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
JSONL = OUT / "bench.jsonl"
PROMPT_LEN = 256
MAX_TOKENS = 128


def dummy_ids(n: int, salt: int = 0) -> list[int]:
    n = max(1, n)
    return [((i * 7919 + 10007 + salt * 97) % 150000) + 100 for i in range(n)]


def _mode(enforce_eager: bool, torch_compile: bool) -> str:
    if torch_compile:
        return "cg+compile" if not enforce_eager else "compile"
    return "cg" if not enforce_eager else "eager"


def make_engine(
    tp: int,
    pp: int,
    model: str = MODEL,
    *,
    enforce_eager: bool = True,
    torch_compile: bool = False,
    device_ids: list[int] | None = None,
    pp_layer_counts: list[int] | None = None,
    num_kvcache_blocks: int = 64,
) -> Engine:
    return Engine(
        EngineConfig(
            model=model,
            tp_size=tp,
            pp_size=pp,
            enforce_eager=enforce_eager,
            torch_compile=torch_compile,
            dtype="bfloat16",
            context_len=4096,
            max_num_seqs=16,
            max_num_batched_tokens=4096,
            prefix_backend="none",
            gpu_memory_utilization=0.6,
            num_kvcache_blocks=num_kvcache_blocks,
            device_ids=device_ids,
            pp_layer_counts=pp_layer_counts,
        )
    )


def _write_jsonl(record: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with JSONL.open("a") as f:
        f.write(json.dumps(record) + "\n")


def run(
    tp: int,
    pp: int,
    tag: str,
    model: str = MODEL,
    *,
    enforce_eager: bool = True,
    torch_compile: bool = False,
    device_ids: list[int] | None = None,
    pp_layer_counts: list[int] | None = None,
    num_kvcache_blocks: int = 64,
) -> None:
    engine = make_engine(
        tp,
        pp,
        model,
        enforce_eager=enforce_eager,
        torch_compile=torch_compile,
        device_ids=device_ids,
        pp_layer_counts=pp_layer_counts,
        num_kvcache_blocks=num_kvcache_blocks,
    )
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
    batch_tps = n / wall
    rows.append(f"{tag} | {b} | - | {batch_tps:.1f}")

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

    _write_jsonl(
        {
            "engine": "v4",
            "model": Path(model).name,
            "tp": tp,
            "pp": pp,
            "mode": _mode(enforce_eager, torch_compile),
            "ttft_ms": round(ttft * 1000, 2),
            "decode_tps": round(decode_tps, 1),
            "batch_tps": round(batch_tps, 1),
        }
    )
    engine.destroy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, required=True)
    p.add_argument("--pp", type=int, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument(
        "--cg",
        action="store_true",
        help="启用 decode full-pipeline CUDA Graph（默认 eager）",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="启用 prefill torch.compile",
    )
    p.add_argument("--device-ids", type=str, default=None, help="逗号分隔设备映射，如 1 或 0,1")
    p.add_argument("--pp-layer-counts", type=str, default=None, help="不均分 PP 层数，如 12,28")
    p.add_argument("--num-kvcache-blocks", type=int, default=64)
    args = p.parse_args()

    device_ids = None
    if args.device_ids is not None:
        device_ids = [int(x) for x in args.device_ids.split(",")]
    pp_layer_counts = None
    if args.pp_layer_counts is not None:
        pp_layer_counts = [int(x) for x in args.pp_layer_counts.split(",")]

    run(
        args.tp,
        args.pp,
        args.tag,
        str(Path(args.model).expanduser()),
        enforce_eager=not args.cg,
        torch_compile=args.compile,
        device_ids=device_ids,
        pp_layer_counts=pp_layer_counts,
        num_kvcache_blocks=args.num_kvcache_blocks,
    )


if __name__ == "__main__":
    main()
