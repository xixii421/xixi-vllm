"""vllm-v1 命令行：下载 / 列表 / 流式推理（支持 prefix backend）。"""

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
    return path if path.exists() else DEFAULT_CACHE_DIR / name


def run_inference(
    model: str,
    *,
    prompt: str = "Hello, my name is",
    max_new_tokens: int = 200,
    temperature: float = 0.9,
    top_p: float = 0.95,
    top_k: int = 20,
    enforce_eager: bool = True,
    prefix_backend: str = "hash",
    block_size: int = 16,
) -> None:
    model_path = _resolve_model_path(model)
    print(f"Loading {model_path} (prefix={prefix_backend}) ...")
    engine = Engine(
        EngineConfig(
            model=str(model_path),
            enforce_eager=enforce_eager,
            prefix_backend=prefix_backend,  # type: ignore[arg-type]
            block_size=block_size,
        )
    )

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
    print(
        f"\n\n{n} tokens in {elapsed:.2f}s ({n / elapsed:.1f} tok/s) "
        f"| cached_prompt_tokens={engine.last_num_cached_tokens}"
    )


def main() -> None:
    names = ", ".join(s.local_name for s in QWEN3_REGISTRY)
    p = argparse.ArgumentParser(description="vllm-v1 Qwen3 + KV / prefix cache")
    p.add_argument("--model", default="Qwen3-0.6B", help=f"dir under {DEFAULT_CACHE_DIR} or path")
    p.add_argument("--prompt", default="Hello, my name is")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--compile", action="store_true", help="启用 torch.compile（默认 eager）")
    p.add_argument(
        "--prefix-backend",
        choices=("none", "hash", "radix"),
        default="hash",
        help="前缀索引：none=不共享 / hash=链式hash / radix=基数树",
    )
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--download", action="store_true", help=f"下载注册表到 {DEFAULT_CACHE_DIR}")
    p.add_argument("--verify-all", action="store_true", help="对显存放得下的模型做冒烟测试")
    p.add_argument("--list", action="store_true", help="列出注册表与按显存筛选结果")
    args = p.parse_args()

    if args.list:
        vram = gpu_memory_gb() or 24.0
        selected = select_models_for_vram(vram_gb=vram)
        print(f"Cache: {DEFAULT_CACHE_DIR}")
        print(f"VRAM:  {vram:.1f} GB")
        print(f"All:   {names}")
        print(f"Fit:   {', '.join(s.local_name for s in selected)}")
        return
    if args.verify_all:
        verify_models(download=True)
        return
    if args.download:
        print(f"Downloading to {DEFAULT_CACHE_DIR} ...")
        download_models()
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
        prefix_backend=args.prefix_backend,
        block_size=args.block_size,
    )


if __name__ == "__main__":
    main()
