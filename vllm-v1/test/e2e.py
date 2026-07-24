"""vllm-v1 端到端测试：greedy 一致性 + 前缀二次命中。

用法:

    cd vllm-v1
    python test/test_e2e.py --model Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import EngineConfig
from src.engine import Engine
from src.model_hub import DEFAULT_CACHE_DIR

_LONG_PROMPT = ("The quick brown fox jumps over the lazy dog. " * 8).strip()
_SHORT_PROMPT = "Hello, say hi in one short sentence."
_BACKENDS = ("none", "hash", "radix")


def _resolve_model(name: str) -> str:
    path = Path(name).expanduser()
    if path.exists():
        return str(path)
    cand = DEFAULT_CACHE_DIR / name
    if cand.exists():
        return str(cand)
    raise FileNotFoundError(f"model not found: {name} (tried {cand})")


def _generate(eng: Engine, prompt: str, max_new: int) -> str:
    return "".join(eng.generate(prompt, max_new_tokens=max_new, temperature=0.0))


def test_greedy_backends_agree(eng: Engine, max_new: int) -> str:
    """冷启动：none / hash / radix 对同一短 prompt 的 greedy 输出应一致。"""
    texts: dict[str, str] = {}
    for backend in _BACKENDS:
        eng.reset_kv(backend)
        texts[backend] = _generate(eng, _SHORT_PROMPT, max_new)
        print(
            f"[cold] backend={backend:5s} cached={eng.last_num_cached_tokens} "
            f"text={texts[backend]!r}"
        )
        assert texts[backend], f"{backend} produced empty text"

    assert texts["none"] == texts["hash"] == texts["radix"], (
        f"greedy mismatch: none={texts['none']!r} "
        f"hash={texts['hash']!r} radix={texts['radix']!r}"
    )
    return texts["none"]


def test_prefix_second_hit(eng: Engine, backend: str, max_new: int) -> None:
    """同一 Engine 连跑两次长 prompt：开启前缀时应命中；none 应仍为 0。"""
    eng.reset_kv(backend)
    block_size = eng.config.block_size
    prompt_len = len(eng.tokenizer.encode(_LONG_PROMPT))
    full_cached = (prompt_len // block_size) * block_size

    t1 = _generate(eng, _LONG_PROMPT, max_new)
    c1 = eng.last_num_cached_tokens
    t2 = _generate(eng, _LONG_PROMPT, max_new)
    c2 = eng.last_num_cached_tokens

    print(
        f"[prefix] backend={backend:5s} prompt_tokens={prompt_len} "
        f"first_cached={c1} second_cached={c2}"
    )
    print(f"         text1={t1!r}")
    print(f"         text2={t2!r}")

    assert t1 == t2, f"{backend}: greedy outputs differ across runs"
    assert t1, f"{backend}: empty generation"

    if backend == "none":
        assert c2 == 0, f"none backend must not prefix-hit, got cached={c2}"
    else:
        assert c2 >= block_size, f"{backend}: expected prefix hit ≥ {block_size}, got {c2}"
        assert c2 <= full_cached, f"{backend}: cached {c2} > full blocks {full_cached}"
        assert c2 > c1 or c1 == 0, (
            f"{backend}: second run should reuse more than a cold miss "
            f"(first={c1}, second={c2})"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="vllm-v1 end-to-end test")
    ap.add_argument("--model", default="Qwen3-0.6B")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=16)
    args = ap.parse_args()

    model = _resolve_model(args.model)
    print(f"model={model}")
    print("loading engine once …")
    eng = Engine(
        EngineConfig(
            model=model,
            context_len=2048,
            enforce_eager=True,
            prefix_backend="hash",
            block_size=args.block_size,
        )
    )

    print("--- 1) cold greedy: none / hash / radix ---")
    test_greedy_backends_agree(eng, args.max_new_tokens)

    print("--- 2) prefix second hit: none / hash / radix ---")
    for backend in _BACKENDS:
        test_prefix_second_hit(eng, backend, args.max_new_tokens)

    print("E2E_OK")


if __name__ == "__main__":
    main()
