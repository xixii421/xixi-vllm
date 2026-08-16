"""引擎端到端功能正确性。

覆盖：
- eager greedy 两次可复现
- decode CUDA Graph 与 eager 输出一致
- torch.compile + CG 与 eager 输出一致
- prompt 超长截断、max_tokens 越界收敛
- max_tokens 终止、真实文本 EOS smoke（不强制命中 EOS，只保证正常终止）

用法:
    python test/test_engine.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import destroy, dummy_ids, greedy_ids, make_engine, warmup  # noqa: E402
from src import SamplingParams  # noqa: E402

PROMPT_LEN = 300  # 跨 block 边界，充分覆盖 decode 打包路径
MAX_TOKENS = 16


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def check_eager():
    engine = make_engine(cg=False, compile=False, prefix="none")
    try:
        prompt = dummy_ids(PROMPT_LEN, salt=1)
        a = greedy_ids(engine, prompt, MAX_TOKENS)
        b = greedy_ids(engine, prompt, MAX_TOKENS)
        _check(a == b, "eager 两次结果不一致")
        _check(len(a) == MAX_TOKENS, "completion 数量不等于 max_tokens")
        return a
    finally:
        destroy(engine)


def check_cg(base):
    engine = make_engine(cg=True, compile=False, prefix="none")
    try:
        warmup(engine, 128)
        out = greedy_ids(engine, dummy_ids(PROMPT_LEN, salt=1), MAX_TOKENS)
        _check(out == base, "decode CG 输出与 eager 不一致")
    finally:
        destroy(engine)


def check_compile(base):
    engine = make_engine(cg=True, compile=True, prefix="none")
    try:
        warmup(engine, 128)
        out = greedy_ids(engine, dummy_ids(PROMPT_LEN, salt=1), MAX_TOKENS)
        _check(out == base, "compile+CG 输出与 eager 不一致")
    finally:
        destroy(engine)


def check_truncate_clamp():
    engine = make_engine(cg=False, compile=False, prefix="none", context_len=64)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seq = engine.add_request(
                dummy_ids(128, salt=9), SamplingParams(temperature=0.0, max_tokens=1)
            )
            _check(seq.num_prompt_tokens == 64, "prompt 未截断到 context_len")
            _check(seq.max_tokens == 0, "截断后 max_tokens 未收敛到 0")

            seq2 = engine.add_request(
                dummy_ids(50, salt=10), SamplingParams(temperature=0.0, max_tokens=100)
            )
            _check(seq2.max_tokens == 14, "max_tokens 未收敛到 context_len - prompt_len")
    finally:
        destroy(engine)


def check_max_tokens_termination():
    engine = make_engine(cg=False, compile=False, prefix="none", context_len=128)
    try:
        sp = SamplingParams(temperature=0.0, max_tokens=5, ignore_eos=True)
        s = engine.add_request(dummy_ids(16, salt=11), sp)
        while not s.is_finished:
            engine.step()
        _check(s.num_completion_tokens == 5, "max_tokens 终止后 completion 数量错误")
    finally:
        destroy(engine)


def check_eos_smoke():
    engine = make_engine(cg=False, compile=False, prefix="none", context_len=256)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = engine.tokenizer.encode("The capital of France is Paris")
            s = engine.add_request(text, SamplingParams(temperature=0.0, max_tokens=64))
        while not s.is_finished:
            engine.step()
        _check(s.num_completion_tokens <= 64, "EOS smoke completion 超出上限")
    finally:
        destroy(engine)


def main() -> None:
    torch.cuda.set_device("cuda:0")

    base = None
    cases = [
        ("eager_reproducible", check_eager),
        ("cg_equiv_eager", check_cg),
        ("compile_equiv_eager", check_compile),
        ("truncate_clamp", check_truncate_clamp),
        ("max_tokens_termination", check_max_tokens_termination),
        ("eos_smoke", check_eos_smoke),
    ]

    failed = 0
    for name, fn in cases:
        try:
            if name in ("cg_equiv_eager", "compile_equiv_eager"):
                if base is None:
                    base = check_eager()
                fn(base)
            else:
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
