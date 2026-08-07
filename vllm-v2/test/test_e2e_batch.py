"""E2E: single stream + generate_batch + prefix hit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import EngineConfig
from src.engine import Engine
from src.model_hub import DEFAULT_CACHE_DIR
from src.sampling_params import SamplingParams


def _resolve(name: str) -> str:
    p = Path(name).expanduser()
    if p.exists():
        return str(p)
    c = DEFAULT_CACHE_DIR / name
    if c.exists():
        return str(c)
    raise FileNotFoundError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-0.6B")
    ap.add_argument("--max-new-tokens", type=int, default=12)
    args = ap.parse_args()
    model = _resolve(args.model)

    eng = Engine(
        EngineConfig(
            model=model,
            prefix_backend="none",
            max_num_seqs=4,
            context_len=512,
        )
    )
    text = "".join(
        eng.generate("Hello,", max_new_tokens=args.max_new_tokens, temperature=0.0)
    )
    print(f"single={text!r}")
    assert text

    eng2 = Engine(
        EngineConfig(
            model=model,
            prefix_backend="hash",
            max_num_seqs=4,
            context_len=512,
            block_size=16,
        )
    )
    prompts = [
        "The quick brown fox jumps over the lazy dog. Cont:",
        "Hello from batch A:",
        "Hello from batch B:",
    ]
    outs = eng2.generate_batch(
        prompts, SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    )
    for p, o in zip(prompts, outs):
        print(f"batch {p!r} → {o!r}")
        assert o

    # Prefix: second identical long prompt should cache full blocks.
    long_p = "The quick brown fox jumps over the lazy dog. " * 3
    _ = "".join(eng2.generate(long_p, max_new_tokens=4, temperature=0.0))
    _ = "".join(eng2.generate(long_p, max_new_tokens=4, temperature=0.0))
    print(f"prefix second cached≈{eng2.last_num_cached_tokens}")
    assert eng2.last_num_cached_tokens > 0
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
