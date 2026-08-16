"""05 — 机制：mix_prefill_decode 开/关（dec@P / short_P）。

4 条短请求（prompt=128, max_tokens=8）先入队，再入 1 条长请求
（prompt=3000, max_tokens=4）；max_num_batched_tokens=1024，长请求 prefill 跨 3 chunk。

- dec@P  : 长请求未完成 prefill 的窗口内，短请求产出的 decode token 数
- short_P: 从长请求入队到全部短请求完成的时间

用法:
    python bench/05_scheduler.py
"""
from __future__ import annotations

import argparse
import time

import torch

from common import destroy, dummy_ids, make_engine, warmup, write_result

from src import SamplingParams  # noqa: E402
from src.memory.sequence import SequenceStatus  # noqa: E402


def run_scenario(engine) -> dict:
    sp = SamplingParams(temperature=0.0, max_tokens=8)
    sp_long = SamplingParams(temperature=0.0, max_tokens=4)
    shorts = [
        engine.add_request(dummy_ids(128, salt=i), sp) for i in range(4)
    ]
    long = engine.add_request(dummy_ids(3000, salt=9), sp_long)

    t0 = time.perf_counter()
    window_started = False
    recorded = False
    base_short = 0
    dec_at_p = 0
    long_ttft = None
    short_p = None

    while not engine.scheduler.is_finished():
        engine.step()
        now = (time.perf_counter() - t0) * 1000
        if not recorded:
            if not window_started and long.num_cached_tokens > 0:
                window_started = True
                base_short = sum(s.num_completion_tokens for s in shorts)
            elif window_started and long.num_cached_tokens >= long.num_prompt_tokens:
                dec_at_p = sum(s.num_completion_tokens for s in shorts) - base_short
                recorded = True
        if long_ttft is None and long.num_completion_tokens >= 1:
            long_ttft = now
        if short_p is None and all(s.status == SequenceStatus.FINISHED for s in shorts):
            short_p = now

    return {
        "dec_at_p": dec_at_p,
        "short_p_ms": short_p or (time.perf_counter() - t0) * 1000,
        "all_ms": (time.perf_counter() - t0) * 1000,
        "long_ttft_ms": long_ttft,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.cuda.set_device(args.device)

    lines = [
        "# 05 scheduler  mix vs prefill-first",
        "",
        "| mix | dec@P | short_P (ms) | long TTFT (ms) | all (ms) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for s in lines:
        print(s)
    print()

    for mix in (True, False):
        engine = make_engine(
            cg=False,
            compile=False,
            prefix="none",
            mix=mix,
            max_num_seqs=8,
            max_batched=1024,
            gpu_mem=0.85,
        )
        warmup(engine, 8)
        r = run_scenario(engine)
        row = (
            f"| {mix} | {r['dec_at_p']} | {r['short_p_ms']:.1f} | "
            f"{r['long_ttft_ms']:.1f} | {r['all_ms']:.1f} |"
        )
        print(row)
        lines.append(row)
        destroy(engine)
    write_result("05_scheduler.md", "\n".join(lines))


if __name__ == "__main__":
    main()
