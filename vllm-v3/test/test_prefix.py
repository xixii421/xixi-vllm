"""前缀缓存功能正确性。

验证 hash / radix 命中（含部分命中触发 remap）后，生成的 token 序列与
无缓存 none 完全一致。greedy 确定性保证正确 KV 时输出必然一致。

用法:
    python test/test_prefix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import destroy, dummy_ids, greedy_ids, make_engine, warmup  # noqa: E402

BLOCK = 256


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def check_full_hit():
    prompt = dummy_ids(1024, salt=0)  # 4 个满 block
    none = make_engine(cg=False, compile=False, prefix="none")
    base = greedy_ids(none, prompt, 16)
    destroy(none)

    for backend in ("hash", "radix"):
        engine = make_engine(cg=False, compile=False, prefix=backend)
        try:
            warmup(engine, 8)
            greedy_ids(engine, prompt, 4)  # seed 写入索引
            out = greedy_ids(engine, prompt, 16)
            cached = engine.last_num_cached_tokens
            _check(cached == 1024, f"{backend} full hit cached 不等于 1024")
            _check(out == base, f"{backend} full hit 输出与 none 不一致")
        finally:
            destroy(engine)


def check_partial_remap():
    prefix = dummy_ids(768, salt=0)  # 3 个满 block
    full_a = prefix + dummy_ids(BLOCK, salt=1)
    full_b = prefix + dummy_ids(BLOCK, salt=2)  # 同 prefix，不同后缀

    none = make_engine(cg=False, compile=False, prefix="none")
    base_b = greedy_ids(none, full_b, 16)
    destroy(none)

    for backend in ("hash", "radix"):
        engine = make_engine(cg=False, compile=False, prefix=backend)
        try:
            warmup(engine, 8)
            greedy_ids(engine, full_a, 4)  # seed 写入 3 block 公共前缀
            out = greedy_ids(engine, full_b, 16)
            cached = engine.last_num_cached_tokens
            _check(cached == 768, f"{backend} partial cached 不等于 768")
            _check(out == base_b, f"{backend} partial hit 输出与 none 不一致")
        finally:
            destroy(engine)


def main() -> None:
    torch.cuda.set_device("cuda:0")

    cases = [
        ("prefix_full_hit", check_full_hit),
        ("prefix_partial_remap", check_partial_remap),
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
