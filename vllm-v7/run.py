"""vllm-v7 entry: v6 TP/PP + ZMQ/Radix/OpenAI-ready config."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src import Engine, EngineConfig, SamplingParams
from src.tools.hub import (
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
    image: str | None = None,
    tp_size: int = 1,
    pp_size: int = 1,
    nccl_port: int = 0,
    nccl_host: str = "127.0.0.1",
    device_ids: list[int] | None = None,
    rpc_backend: str = "zmq",
    enable_radix: bool = True,
) -> None:
    model_path = _resolve_model_path(model)
    print(f"Loading {model_path} (tp={tp_size}, pp={pp_size}, rpc={rpc_backend}) ...")
    engine = Engine(
        EngineConfig(
            model=str(model_path),
            enforce_eager=enforce_eager,
            max_num_seqs=8,
            tp_size=tp_size,
            pp_size=pp_size,
            nccl_port=nccl_port,
            nccl_host=nccl_host,
            device_ids=device_ids,
            rpc_backend=rpc_backend,
            enable_radix=enable_radix,
        )
    )

    images = None
    if image:
        from PIL import Image

        images = [Image.open(Path(image).expanduser()).convert("RGB")]
        print(f"Image: {image}")

    print(f"Prompt: {prompt}\n{prompt}", end="", flush=True)
    start = time.perf_counter()
    n = 0
    for tok in engine.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        images=images,
    ):
        print(tok, end="", flush=True)
        n += 1

    elapsed = time.perf_counter() - start
    print(f"\n\n{n} tokens in {elapsed:.2f}s ({n / elapsed:.1f} tok/s)")
    engine.destroy()


def run_batch(
    model: str,
    *,
    max_new_tokens: int = 32,
    enforce_eager: bool = False,
    tp_size: int = 1,
    pp_size: int = 1,
    nccl_port: int = 0,
    nccl_host: str = "127.0.0.1",
    device_ids: list[int] | None = None,
    rpc_backend: str = "zmq",
    enable_radix: bool = True,
) -> None:
    model_path = _resolve_model_path(model)
    print(f"Loading {model_path} for continuous batching ...")
    engine = Engine(
        EngineConfig(
            model=str(model_path),
            enforce_eager=enforce_eager,
            max_num_seqs=8,
            tp_size=tp_size,
            pp_size=pp_size,
            nccl_port=nccl_port,
            nccl_host=nccl_host,
            device_ids=device_ids,
            rpc_backend=rpc_backend,
            enable_radix=enable_radix,
        )
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
    engine.destroy()


def main() -> None:
    names = ", ".join(s.local_name for s in QWEN3_REGISTRY)
    parser = argparse.ArgumentParser(
        description="vllm-v7: ZMQ + Radix + TP/PP + OpenAI-ready"
    )
    parser.add_argument(
        "--model",
        default="Qwen3-0.6B",
        help=f"local dir under {DEFAULT_CACHE_DIR} or absolute path",
    )
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--image", default=None, help="path to an image (Qwen3-VL)")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="enable Decode CUDA Graph (disable enforce_eager)",
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        help="force enforce_eager (disable Decode CUDA Graph)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="run a small continuous-batching demo on 4 prompts",
    )
    parser.add_argument("--tp-size", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--pp-size", type=int, default=1, help="pipeline parallel size")
    parser.add_argument(
        "--nccl-port",
        type=int,
        default=0,
        help="NCCL rendezvous TCP port; 0 = auto-pick a free port",
    )
    parser.add_argument(
        "--device-ids",
        default=None,
        help="comma-separated CUDA device ids for ranks (e.g. 0,1)",
    )
    parser.add_argument(
        "--rpc-backend",
        choices=["zmq", "pipe"],
        default="zmq",
        help="worker RPC: zmq (ullm-style) or pipe (v6)",
    )
    parser.add_argument(
        "--nccl-host",
        default="127.0.0.1",
        help="NCCL rendezvous host (multi-node: master IP)",
    )
    parser.add_argument(
        "--no-radix",
        action="store_true",
        help="disable Radix prefix cache (use flat hash)",
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

    enforce_eager = True if args.eager else (not args.compile)
    device_ids = (
        [int(x) for x in args.device_ids.split(",")] if args.device_ids else None
    )

    if args.batch:
        run_batch(
            args.model,
            max_new_tokens=min(args.max_new_tokens, 64),
            enforce_eager=enforce_eager,
            tp_size=args.tp_size,
            pp_size=args.pp_size,
            nccl_port=args.nccl_port,
            nccl_host=args.nccl_host,
            device_ids=device_ids,
            rpc_backend=args.rpc_backend,
            enable_radix=not args.no_radix,
        )
        return

    run_inference(
        args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        enforce_eager=enforce_eager,
        image=args.image,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        nccl_port=args.nccl_port,
        nccl_host=args.nccl_host,
        device_ids=device_ids,
        rpc_backend=args.rpc_backend,
        enable_radix=not args.no_radix,
    )


if __name__ == "__main__":
    main()
