"""官方 vLLM 基准：与 v4 bench_parallel.py 对齐的指标与场景。

指标：单请求 TTFT（prompt=256）、单请求 decode tok/s、batch=8 聚合吞吐。
维度：tp=1/2 × eager/CG × compile（vLLM V1 已移除 PP，故无 PP 档）。

大模型支持：--model、--cuda-devices（在导入 vllm 前设置 CUDA_VISIBLE_DEVICES）、--jsonl。

用法（在 vllm-v4/ 下，用官方环境 .venv-vllm）:
    ../.venv-vllm/bin/python bench/bench_vllm_official.py --tp 1 --cuda-devices 1 --tag tp1-eager
    ../.venv-vllm/bin/python bench/bench_vllm_official.py --model ~/shared/models/Qwen3-8B --tp 1 --cuda-devices 1 --cg --tag 8b-tp1-cg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL = str(Path.home() / "huggingface" / "Qwen3-0.6B")
OUT = Path(__file__).resolve().parent / "out"
JSONL = OUT / "bench.jsonl"
PROMPT_LEN = 256
MAX_TOKENS = 128


def dummy_ids(n: int, salt: int = 0) -> list[int]:
    n = max(1, n)
    return [((i * 7919 + 10007 + salt * 97) % 150000) + 100 for i in range(n)]


def _mode(cg: bool, compile_mode: int) -> str:
    if compile_mode > 0:
        return "cg+compile" if cg else "compile"
    return "cg" if cg else "eager"


def _import_vllm():
    # 延迟导入，确保 CUDA_VISIBLE_DEVICES 已在 main 中设置。
    from vllm import LLM, SamplingParams  # noqa: E402

    return LLM, SamplingParams


def make_llm(LLM, tp: int, model: str, *, cg: bool, compile_mode: int):
    kwargs: dict = dict(
        model=model,
        tensor_parallel_size=tp,
        enforce_eager=not cg,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=16,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )
    if compile_mode > 0:
        kwargs["compilation_config"] = {"mode": compile_mode}
    return LLM(**kwargs)


def run(tp: int, tag: str, model: str = MODEL, *, cg: bool, compile_mode: int) -> None:
    print(f"=== {tag}: tp={tp} cg={cg} compile={compile_mode} ===", flush=True)
    LLM, SamplingParams = _import_vllm()
    llm = make_llm(LLM, tp, model, cg=cg, compile_mode=compile_mode)
    rows: list[str] = []

    # 预热
    for it in range(3):
        llm.generate(
            prompts=[{"prompt_token_ids": dummy_ids(PROMPT_LEN, salt=it)}],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True),
        )
    torch.cuda.synchronize()

    # 单请求 TTFT：max_tokens=1 的整段耗时 ≈ prefill + 首个 decode step。
    t0 = time.perf_counter()
    llm.generate(
        prompts=[{"prompt_token_ids": dummy_ids(PROMPT_LEN, salt=100)}],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
    )
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0

    # 单请求 decode tok/s
    t0 = time.perf_counter()
    llm.generate(
        prompts=[{"prompt_token_ids": dummy_ids(PROMPT_LEN, salt=200)}],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True),
    )
    torch.cuda.synchronize()
    e2e = time.perf_counter() - t0
    decode_tps = (MAX_TOKENS - 1) / (e2e - ttft) if e2e > ttft else 0.0
    rows.append(f"{tag} | 1 | {ttft * 1000:.2f} | {decode_tps:.1f}")

    # batch=8 聚合吞吐
    b = 8
    prompts = [{"prompt_token_ids": dummy_ids(PROMPT_LEN, salt=1000 + i)} for i in range(b)]
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(prompts=prompts, sampling_params=sp)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    batch_tps = n / wall
    rows.append(f"{tag} | {b} | - | {batch_tps:.1f}")

    header = "config | batch | TTFT (ms) | decode tok/s"
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "bench_vllm_official.txt"
    first = not path.exists() or path.stat().st_size == 0
    with path.open("a") as f:
        if first:
            f.write(header + "\n")
        for r in rows:
            f.write(r + "\n")
            print(r)
    print(f"→ {path}", flush=True)

    with JSONL.open("a") as f:
        f.write(
            json.dumps(
                {
                    "engine": "vllm",
                    "model": Path(model).name,
                    "tp": tp,
                    "pp": 1,
                    "mode": _mode(cg, compile_mode),
                    "ttft_ms": round(ttft * 1000, 2),
                    "decode_tps": round(decode_tps, 1),
                    "batch_tps": round(batch_tps, 1),
                }
            )
            + "\n"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--tag", default="vllm")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--cg", action="store_true", help="禁用 eager（启用 CUDA Graph）")
    p.add_argument("--compile", action="store_true", help="启用 VLLM_COMPILE (mode=3)")
    p.add_argument("--cuda-devices", type=str, default=None, help="CUDA_VISIBLE_DEVICES，如 1 或 0,1")
    args = p.parse_args()

    if args.cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    run(
        args.tp,
        args.tag,
        str(Path(args.model).expanduser()),
        cg=args.cg,
        compile_mode=3 if args.compile else 0,
    )


if __name__ == "__main__":
    main()
