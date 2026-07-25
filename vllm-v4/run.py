"""vllm-v3 entry: FlashInfer + CUDA Graph + continuous batching."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src import Engine, EngineConfig, SamplingParams
from src.models.hub import (
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
    enforce_eager: bool = False,
) -> None:
    model_path = _resolve_model_path(model)
    print(f"Loading {model_path} ...")
    engine = Engine(
        EngineConfig(model=str(model_path), enforce_eager=enforce_eager, max_num_seqs=8)
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
    print(f"\n\n{n} tokens in {elapsed:.2f}s ({n / elapsed:.1f} tok/s)")


def run_image(
    model: str,
    *,
    image: str,
    prompt: str = "Describe this image.",
    max_new_tokens: int = 128,
    enforce_eager: bool = False,
    enable_vit_cg: bool = False,
) -> None:
    from PIL import Image

    model_path = _resolve_model_path(model)
    print(f"Loading multimodal {model_path} ...")
    engine = Engine(
        EngineConfig(
            model=str(model_path),
            enforce_eager=enforce_eager,
            enable_vit_cuda_graph=enable_vit_cg,
            max_num_seqs=8,
        )
    )
    if not engine.config.is_multimodal:
        raise SystemExit(f"{model_path} is not a multimodal (VL) model")

    img = Image.open(image).convert("RGB")
    print(f"Image: {image}\nPrompt: {prompt}\n")
    start = time.perf_counter()
    n = 0
    for tok in engine.generate(
        prompt, max_new_tokens=max_new_tokens, temperature=0.0, images=[img]
    ):
        print(tok, end="", flush=True)
        n += 1
    elapsed = time.perf_counter() - start
    print(f"\n\n{n} tokens in {elapsed:.2f}s ({n / max(elapsed, 1e-9):.1f} tok/s)")


def run_batch(
    model: str,
    *,
    max_new_tokens: int = 32,
    enforce_eager: bool = False,
) -> None:
    model_path = _resolve_model_path(model)
    print(f"Loading {model_path} for continuous batching ...")
    engine = Engine(
        EngineConfig(model=str(model_path), enforce_eager=enforce_eager, max_num_seqs=8)
    )
    prompts = [
        "Hello, my name is",
        "Once upon a time",
        "The capital of France is",
        "In machine learning,",
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    start = time.perf_counter()
    outs = engine.generate_batch(prompts, sp)
    elapsed = time.perf_counter() - start
    for p, o in zip(prompts, outs):
        print(f"\n>>> {p}\n{o}")
    total_tok = sum(len(engine.tokenizer.encode(o)) for o in outs)
    print(f"\nBatch: {len(prompts)} reqs, ~{total_tok} toks in {elapsed:.2f}s")


def main() -> None:
    names = ", ".join(s.local_name for s in QWEN3_REGISTRY)
    parser = argparse.ArgumentParser(
        description="vllm-v3 Qwen3 inference (FlashInfer + CUDA Graph + Continuous Batching)"
    )
    parser.add_argument(
        "--model",
        default="Qwen3-0.6B",
        help=f"local dir under {DEFAULT_CACHE_DIR} or absolute path",
    )
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="disable enforce_eager to enable Decode CUDA Graph",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="run a small continuous-batching demo on 4 prompts",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="path to an image → run the Qwen3-VL multimodal demo",
    )
    parser.add_argument(
        "--enable-vit-cg",
        action="store_true",
        help="capture budget-based ViT CUDA Graphs (multimodal only)",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--verify-all", action="store_true")
    parser.add_argument("--list", action="store_true")
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
        print(f"Downloading to {DEFAULT_CACHE_DIR} ...")
        download_models(list(QWEN3_REGISTRY))
        print("Done.")
        return

    if args.image:
        run_image(
            args.model,
            image=args.image,
            prompt=args.prompt if args.prompt != "Hello, my name is" else "Describe this image.",
            max_new_tokens=args.max_new_tokens,
            enforce_eager=not args.compile,
            enable_vit_cg=args.enable_vit_cg,
        )
        return

    if args.batch:
        run_batch(
            args.model,
            max_new_tokens=min(args.max_new_tokens, 64),
            enforce_eager=not args.compile,
        )
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
