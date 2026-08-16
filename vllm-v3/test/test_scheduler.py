"""调度器功能正确性。

4 条短请求（prompt=128, max_tokens=8）先入队，再入 1 条长请求
（prompt=3000, max_tokens=4），max_num_batched_tokens=1024，长请求 prefill
跨 3 个 chunk。验证 mix 开/关时 dec@P 行为与请求最终完成数量。

用法:
    python test/test_scheduler.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import destroy, dummy_ids, make_engine, warmup  # noqa: E402
from src import SamplingParams  # noqa: E402


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _dec_at_p(engine, shorts, long) -> int:
    """长请求完成 prefill 前，短请求新产出的 decode token 数。完整 drain 后再返回。"""
    window_started = False
    recorded = False
    base = 0
    dec = 0
    while not engine.scheduler.is_finished():
        engine.step()
        if not recorded:
            if not window_started and long.num_cached_tokens > 0:
                window_started = True
                base = sum(s.num_completion_tokens for s in shorts)
            elif window_started and long.num_cached_tokens >= long.num_prompt_tokens:
                dec = sum(s.num_completion_tokens for s in shorts) - base
                recorded = True
    return dec


def _run(mix: bool) -> tuple[int, list, object]:
    engine = make_engine(
        cg=False, compile=False, prefix="none", mix=mix,
        max_num_seqs=8, max_batched=1024, gpu_mem=0.85,
    )
    try:
        warmup(engine, 8)
        sp = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
        sp_long = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
        shorts = [engine.add_request(dummy_ids(128, salt=i), sp) for i in range(4)]
        long = engine.add_request(dummy_ids(3000, salt=9), sp_long)
        dec = _dec_at_p(engine, shorts, long)
        return dec, shorts, long
    finally:
        destroy(engine)


def check_mix_on():
    dec, shorts, long = _run(True)
    _check(dec > 0, "mix=True 时长请求 prefill 窗口内短请求未产出 decode")
    _check(all(s.num_completion_tokens == 8 for s in shorts), "短请求未全部完成")
    _check(long.num_completion_tokens == 4, "长请求未完成")


def check_mix_off():
    dec, shorts, long = _run(False)
    _check(dec == 0, "mix=False 时 prefill 优先，短请求不应在 prefill 窗口 decode")
    _check(all(s.num_completion_tokens == 8 for s in shorts), "短请求未全部完成")
    _check(long.num_completion_tokens == 4, "长请求未完成")


def main() -> None:
    torch.cuda.set_device("cuda:0")

    cases = [
        ("scheduler_mix_on", check_mix_on),
        ("scheduler_mix_off", check_mix_off),
    ]
    failed = 0
    for name, fn in cases:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {e}")

    if failed:
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
