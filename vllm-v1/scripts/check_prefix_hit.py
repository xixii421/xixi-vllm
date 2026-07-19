#!/usr/bin/env python3
"""同一长 prompt 连跑两次，断言第二次前缀命中且 greedy 文本一致。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import EngineConfig
from src.engine import Engine
from src.model_hub import DEFAULT_CACHE_DIR


def _resolve(model: str) -> Path:
    p = Path(model).expanduser()
    return p if p.exists() else DEFAULT_CACHE_DIR / model


def run_once(engine: Engine, prompt: str, n: int) -> str:
    return "".join(
        engine.generate(prompt, max_new_tokens=n, temperature=0.0)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-0.6B")
    ap.add_argument("--backend", choices=("hash", "radix", "none"), default="hash")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()

    model = str(_resolve(args.model))
    # ≥ 2 full blocks so a hit is observable
    prompt = ("The quick brown fox jumps over the lazy dog. " * 8).strip()

    eng = Engine(
        EngineConfig(
            model=model,
            enforce_eager=True,
            prefix_backend=args.backend,
            block_size=args.block_size,
            context_len=2048,
        )
    )

    t1 = run_once(eng, prompt, args.max_new_tokens)
    c1 = eng.last_num_cached_tokens
    t2 = run_once(eng, prompt, args.max_new_tokens)
    c2 = eng.last_num_cached_tokens

    print(f"backend={args.backend} block_size={args.block_size}")
    print(f"prompt_tokens≈{len(eng.tokenizer.encode(prompt))} first_cached={c1} second_cached={c2}")
    print(f"text1={t1!r}")
    print(f"text2={t2!r}")

    if t1 != t2:
        raise SystemExit("FAIL: greedy outputs differ between runs")
    if args.backend == "none":
        if c2 != 0:
            raise SystemExit(f"FAIL: none backend should not cache, got {c2}")
    else:
        full = (len(eng.tokenizer.encode(prompt)) // args.block_size) * args.block_size
        if c2 < args.block_size:
            raise SystemExit(f"FAIL: expected prefix hit ≥ block_size, got {c2}")
        if c2 > full:
            raise SystemExit(f"FAIL: cached {c2} > full blocks {full}")
    print("OK")


if __name__ == "__main__":
    main()
