"""官方 vLLM 对照基准（单卡与 TP）。

官方 V1 离线路径不支持 PP，也不支持 v4 式多 Engine DP。
指标与 bench_v4 的 dp=1 对齐：TTFT、decode、batch8。

用法（需官方 vLLM 环境）:
  python bench/bench_vllm.py --tp 1 --cg
  python bench/bench_vllm.py --tp 2 --cuda-devices 0,1 --compile
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
BENCH = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from common import (  # noqa: E402
    BATCH_SIZE,
    DEFAULT_MODEL,
    MAX_TOKENS,
    PROMPT_LEN,
    append_jsonl,
    base_record,
    dummy_ids,
    mode_name,
)


def make_llm(*, model: str, tp: int, cg: bool, compile: bool):
    from vllm import LLM

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
    if compile:
        kwargs["compilation_config"] = {"mode": 3}
    return LLM(**kwargs)


def warmup(llm, SamplingParams, *, rounds: int = 3) -> None:
    for it in range(rounds):
        llm.generate(
            prompts=[{"prompt_token_ids": dummy_ids(PROMPT_LEN, salt=it)}],
            sampling_params=SamplingParams(
                temperature=0.0, max_tokens=4, ignore_eos=True
            ),
        )
    torch.cuda.synchronize()


def measure(llm, SamplingParams) -> tuple[float, float, float]:
    """返回 (ttft_ms, decode_tps, batch_tps)。

    TTFT 用 max_tokens=1 的整段墙钟近似 prefill 加首 decode。
    """
    t0 = time.perf_counter()
    llm.generate(
        prompts=[{"prompt_token_ids": dummy_ids(PROMPT_LEN, salt=100)}],
        sampling_params=SamplingParams(
            temperature=0.0, max_tokens=1, ignore_eos=True
        ),
    )
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0

    t0 = time.perf_counter()
    llm.generate(
        prompts=[{"prompt_token_ids": dummy_ids(PROMPT_LEN, salt=200)}],
        sampling_params=SamplingParams(
            temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True
        ),
    )
    torch.cuda.synchronize()
    e2e = time.perf_counter() - t0
    decode_tps = (MAX_TOKENS - 1) / (e2e - ttft) if e2e > ttft else 0.0

    prompts = [
        {"prompt_token_ids": dummy_ids(PROMPT_LEN, salt=1000 + i)}
        for i in range(BATCH_SIZE)
    ]
    sp = SamplingParams(
        temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(prompts=prompts, sampling_params=sp)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    batch_tps = n / wall if wall > 0 else 0.0
    return ttft * 1000.0, decode_tps, batch_tps


def run(args: argparse.Namespace) -> int:
    if args.pp > 1:
        print("官方 vLLM V1 离线路径不支持 PP，请只用 --tp", flush=True)
        return 2
    if args.dp > 1:
        print("官方对照不做 v4 式 DP 副本，请用 bench_v4.py", flush=True)
        return 2

    model = str(Path(args.model).expanduser())
    cg = bool(args.cg)
    compile = bool(args.compile)
    mode = mode_name(cg=cg, compile=compile)
    tag = args.tag or f"vllm-{Path(model).name}-tp{args.tp}-{mode}"
    record = base_record(
        engine="vllm",
        model=model,
        tp=args.tp,
        pp=1,
        dp=1,
        mode=mode,
        tag=tag,
    )
    print(f"=== {tag}: tp={args.tp} mode={mode} ===", flush=True)

    try:
        from vllm import SamplingParams

        llm = make_llm(model=model, tp=args.tp, cg=cg, compile=compile)
        warmup(llm, SamplingParams)
        ttft_ms, decode_tps, batch_tps = measure(llm, SamplingParams)
        record["ttft_ms"] = round(ttft_ms, 2)
        record["decode_tps"] = round(decode_tps, 1)
        record["batch_tps"] = round(batch_tps, 1)
        print(
            f"{tag} | ttft={record['ttft_ms']:.2f}ms "
            f"| decode={record['decode_tps']:.1f} "
            f"| batch8={record['batch_tps']:.1f}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"FAIL {tag}: {record['error']}", flush=True)
        append_jsonl(record)
        return 1

    append_jsonl(record)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="official vLLM bench")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1, help="仅允许 1")
    p.add_argument("--dp", type=int, default=1, help="仅允许 1")
    p.add_argument("--cg", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--tag", default=None)
    p.add_argument("--cuda-devices", type=str, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
