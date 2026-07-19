"""vllm-v2 CLI：单流 / 批处理。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.config import EngineConfig
from src.engine import Engine
from src.model_hub import DEFAULT_CACHE_DIR
from src.sampling_params import SamplingParams


def _resolve_model(name: str) -> Path:
    path = Path(name).expanduser()
    return path if path.exists() else DEFAULT_CACHE_DIR / name


def main() -> None:
    p = argparse.ArgumentParser(description="vllm-v2 Continuous Batching")
    p.add_argument("--model", default="Qwen3-0.6B")
    p.add_argument("--prompt", default="Hello, my name is")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--prefix-backend", choices=("none", "hash", "radix"), default="hash")
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--batch", action="store_true", help="run generate_batch demo")
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    model = str(_resolve_model(args.model))
    eng = Engine(
        EngineConfig(
            model=model,
            enforce_eager=not args.compile,
            prefix_backend=args.prefix_backend,  # type: ignore[arg-type]
            max_num_seqs=args.max_num_seqs,
        )
    )

    if args.batch:
        prompts = [
            "Hello, my name is",
            "The capital of France is",
            "In one sentence, AI is",
            "Write a short greeting:",
        ]
        sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)
        t0 = time.perf_counter()
        outs = eng.generate_batch(prompts, sp)
        dt = time.perf_counter() - t0
        for i, (prompt, text) in enumerate(zip(prompts, outs)):
            print(f"[{i}] {prompt!r} → {text!r}")
        print(f"\n{len(prompts)} requests in {dt:.2f}s")
        return

    print(args.prompt, end="", flush=True)
    t0 = time.perf_counter()
    n = 0
    for tok in eng.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    ):
        print(tok, end="", flush=True)
        n += 1
    dt = time.perf_counter() - t0
    print(f"\n\n{n} tokens in {dt:.2f}s ({n / dt:.1f} tok/s)")


if __name__ == "__main__":
    main()
