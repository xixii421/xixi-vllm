"""vllm-v0 entry: single-model inference or multi-model verification."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.config import EngineConfig
from src.engine import Engine
from src.model_hub import (
    DEFAULT_CACHE_DIR,
    QWEN3_REGISTRY,
    download_models,
    gpu_memory_gb,
    select_models_for_vram,
    verify_models,
)


def _resolve_model_path(name: str) -> Path:
    path = Path(name).expanduser()
    if path.exists():
        return path
    return DEFAULT_CACHE_DIR / name


def run_inference(
    model: str,
    *,
    prompt: str = "Hello, my name is",
    max_new_tokens: int = 200,
    temperature: float = 0.9,
    top_p: float = 0.95,
    top_k: int = 20,
    enforce_eager: bool = True,
) -> None:
    model_path = _resolve_model_path(model)
    print(f"Loading {model_path} ...")
    engine = Engine(EngineConfig(model=str(model_path), enforce_eager=enforce_eager))

    print(f"Prompt: {prompt}\n{prompt}", end="", flush=True)
    start = time.perf_counter()
    n = 0
    for tok in engine.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    ):
        print(tok, end="", flush=True)
        n += 1

    elapsed = time.perf_counter() - start
    print(f"\n\n{n} tokens in {elapsed:.2f}s ({n / elapsed:.1f} tok/s)")


def main() -> None:
    names = ", ".join(s.local_name for s in QWEN3_REGISTRY)
    parser = argparse.ArgumentParser(description="vllm-v0 Qwen3 inference and model verification")
    parser.add_argument(
        "--model",
        default="Qwen3-0.6B",
        help=f"local dir name under {DEFAULT_CACHE_DIR} or absolute path (default: Qwen3-0.6B)",
    )
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--compile", action="store_true", help="enable torch.compile (default: eager)")
    parser.add_argument(
        "--download",
        action="store_true",
        help=f"download Qwen3 weights to {DEFAULT_CACHE_DIR}",
    )
    parser.add_argument(
        "--verify-all",
        action="store_true",
        help="download (if needed) and smoke-test all models that fit in GPU VRAM",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list registered models and VRAM budget selection",
    )
    args = parser.parse_args()

    if args.list:
        vram = gpu_memory_gb() or 24.0
        selected = select_models_for_vram(vram_gb=vram)
        print(f"Cache dir: {DEFAULT_CACHE_DIR}")
        print(f"GPU VRAM:  {vram:.1f} GB")
        print(f"Registry:  {names}")
        print(f"Selected for verify-all: {', '.join(s.local_name for s in selected)}")
        return

    if args.verify_all:
        verify_models(download=True)
        return

    if args.download:
        specs = list(QWEN3_REGISTRY)
        print(f"Downloading to {DEFAULT_CACHE_DIR} ...")
        download_models(specs)
        print("Done.")
        return

    run_inference(
        args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        enforce_eager=not args.compile,
    )


if __name__ == "__main__":
    main()
