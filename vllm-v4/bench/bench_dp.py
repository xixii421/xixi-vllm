"""v4 DP 吞吐对比：dp=1（单卡）vs dp=2（两卡）的聚合吞吐，及 CG/compile 开关。

测量口径：把 N 条等长请求一次性加入（dp>1 时门面 round-robin 分发），
从第一次 step 计时到全部完成，聚合吞吐 = 总生成 token 数 / 墙钟时间。

大模型支持：--model、--device-ids。
结果同时写 bench/out/bench_dp.txt 与 bench/out/bench.jsonl（供绘图）。

用法（在 vllm-v4/ 下）:
    ../.venv2/bin/python bench/bench_dp.py --model ~/huggingface/Qwen3-4B --dp 1 --n-reqs 32 --tag "dp1-eager"
    ../.venv2/bin/python bench/bench_dp.py --model ~/huggingface/Qwen3-4B --dp 2 --n-reqs 32 --cg --tag "dp2-cg"
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
PROMPT_LEN = 128
MAX_TOKENS = 128


def dummy_ids(n: int, salt: int = 0) -> list[int]:
    n = max(1, n)
    return [((i * 7919 + 10007 + salt * 97) % 150000) + 100 for i in range(n)]


def _mode(cg: bool, compile: bool) -> str:
    if compile:
        return "cg+compile" if cg else "compile"
    return "cg" if cg else "eager"


def make_engine(
    dp: int,
    model: str = MODEL,
    *,
    cg: bool = False,
    compile: bool = False,
    device_ids: list[int] | None = None,
    num_kvcache_blocks: int = 128,
    max_num_seqs: int = 64,
) -> Engine:
    return Engine(
        EngineConfig(
            model=model,
            tp_size=1,
            pp_size=1,
            dp_size=dp,
            enforce_eager=not cg,
            torch_compile=compile,
            dtype="bfloat16",
            context_len=4096,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=8192,
            prefix_backend="none",
            gpu_memory_utilization=0.6,
            num_kvcache_blocks=num_kvcache_blocks,
            device_ids=device_ids,
        )
    )


def run(
    dp: int,
    n_reqs: int,
    tag: str,
    model: str = MODEL,
    *,
    cg: bool,
    compile: bool,
    device_ids: list[int] | None = None,
    num_kvcache_blocks: int = 128,
    max_num_seqs: int = 64,
) -> None:
    engine = make_engine(
        dp,
        model,
        cg=cg,
        compile=compile,
        device_ids=device_ids,
        num_kvcache_blocks=num_kvcache_blocks,
        max_num_seqs=max_num_seqs,
    )

    # 预热：compile + CG 录图不计时。
    sp_w = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
    for it in range(2):
        for i in range(n_reqs):
            engine.add_request(dummy_ids(PROMPT_LEN, salt=it * 13 + i), sp_w)
        while engine.has_unfinished():
            engine.step()
    torch.cuda.synchronize()

    sp = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=MAX_TOKENS)
    seqs = [
        engine.add_request(dummy_ids(PROMPT_LEN, salt=1000 + i), sp)
        for i in range(n_reqs)
    ]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while engine.has_unfinished():
        engine.step()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    n_tok = sum(s.num_completion_tokens for s in seqs)
    tps = n_tok / wall
    print(f"{tag} | dp={dp} | reqs={n_reqs} | {n_tok} tok in {wall:.3f}s "
          f"| {tps:.1f} tok/s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "bench_dp.txt"
    with path.open("a") as f:
        f.write(f"{tag} | dp={dp} | {n_reqs} | {tps:.1f}\n")

    with JSONL.open("a") as f:
        f.write(
            json.dumps(
                {
                    "engine": "v4",
                    "model": Path(model).name,
                    "dp": dp,
                    "mode": _mode(cg, compile),
                    "n_reqs": n_reqs,
                    "throughput_tps": round(tps, 1),
                }
            )
            + "\n"
        )
    engine.destroy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dp", type=int, required=True)
    p.add_argument("--n-reqs", type=int, default=32)
    p.add_argument("--tag", required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--cg", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--device-ids", type=str, default=None, help="逗号分隔设备映射，如 1 或 0,1")
    p.add_argument("--num-kvcache-blocks", type=int, default=128)
    p.add_argument("--max-num-seqs", type=int, default=64, help="CUDA Graph 捕获最大 batch（小值可降图内存）")
    args = p.parse_args()

    device_ids = None
    if args.device_ids is not None:
        device_ids = [int(x) for x in args.device_ids.split(",")]

    run(
        args.dp,
        args.n_reqs,
        args.tag,
        str(Path(args.model).expanduser()),
        cg=args.cg,
        compile=args.compile,
        device_ids=device_ids,
        num_kvcache_blocks=args.num_kvcache_blocks,
        max_num_seqs=args.max_num_seqs,
    )


if __name__ == "__main__":
    main()
