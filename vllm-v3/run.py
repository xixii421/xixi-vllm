"""vllm-v3 统一入口 — FlashInfer + CUDA Graph + Continuous Batching。

用法:
    python run.py                                    # 单条推理（使用 config.yaml）
    python run.py --cg                               # 启用 Decode CUDA Graph
    python run.py --batch                            # 批量推理演示
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from src import Engine, EngineConfig, SamplingParams
from src.models.hub import (
    DEFAULT_CACHE_DIR,
    QWEN3_REGISTRY,
    download_models,
    gpu_memory_gb,
    select_models_for_vram,
)

# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent

DEFAULTS = {
    "model": "Qwen3-0.6B",
    "prompt": "Hello, my name is",
    "max_new_tokens": 200,
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 20,
    "enforce_eager": True,
    "prefill_cuda_graph": False,
    "dtype": "auto",
    "context_len": 4096,
    "max_num_seqs": 256,
    "max_num_batched_tokens": 8192,
    "mix_prefill_decode": True,
    "kvcache_block_size": 256,
    "prefix_backend": "hash",
    "gpu_memory_utilization": 0.9,
    "num_kvcache_blocks": 0,
    "flashinfer_use_tensor_cores": True,
    "flashinfer_non_blocking": True,
}


def load_config(path: Path) -> dict:
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    return {k: cfg.get(k, v) for k, v in DEFAULTS.items()}


# ---------------------------------------------------------------------------
# 核心逻辑
# ---------------------------------------------------------------------------


def _resolve_model(name: str) -> Path:
    path = Path(name).expanduser()
    return path if path.exists() else DEFAULT_CACHE_DIR / name


def run_single(cfg: dict) -> None:
    model_path = _resolve_model(cfg["model"])
    print(f"Loading {model_path} ...")
    engine = Engine(
        EngineConfig(
            model=str(model_path),
            enforce_eager=cfg["enforce_eager"],
            prefill_cuda_graph=cfg["prefill_cuda_graph"],
            dtype=cfg["dtype"],
            context_len=cfg["context_len"],
            max_num_seqs=cfg["max_num_seqs"],
            max_num_batched_tokens=cfg["max_num_batched_tokens"],
            mix_prefill_decode=cfg["mix_prefill_decode"],
            kvcache_block_size=cfg["kvcache_block_size"],
            prefix_backend=cfg["prefix_backend"],
            gpu_memory_utilization=cfg["gpu_memory_utilization"],
            num_kvcache_blocks=cfg["num_kvcache_blocks"],
            flashinfer_use_tensor_cores=cfg["flashinfer_use_tensor_cores"],
            flashinfer_non_blocking=cfg["flashinfer_non_blocking"],
        )
    )

    prompt = cfg["prompt"]
    print(f"Prompt: {prompt}\n{prompt}", end="", flush=True)
    start = time.perf_counter()
    n = 0
    for tok in engine.generate(
        prompt,
        max_new_tokens=cfg["max_new_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
    ):
        print(tok, end="", flush=True)
        n += 1
    elapsed = time.perf_counter() - start
    print(f"\n\n{n} tokens in {elapsed:.2f}s ({n / elapsed:.1f} tok/s)")


def run_batch(cfg: dict) -> None:
    model_path = _resolve_model(cfg["model"])
    print(f"Loading {model_path} for continuous batching ...")
    engine = Engine(
        EngineConfig(
            model=str(model_path),
            enforce_eager=cfg["enforce_eager"],
            prefill_cuda_graph=cfg["prefill_cuda_graph"],
            dtype=cfg["dtype"],
            context_len=cfg["context_len"],
            max_num_seqs=cfg["max_num_seqs"],
            max_num_batched_tokens=cfg["max_num_batched_tokens"],
            mix_prefill_decode=cfg["mix_prefill_decode"],
            kvcache_block_size=cfg["kvcache_block_size"],
            prefix_backend=cfg["prefix_backend"],
            gpu_memory_utilization=cfg["gpu_memory_utilization"],
            num_kvcache_blocks=cfg["num_kvcache_blocks"],
            flashinfer_use_tensor_cores=cfg["flashinfer_use_tensor_cores"],
            flashinfer_non_blocking=cfg["flashinfer_non_blocking"],
        )
    )

    prompts = [
        "Hello, my name is",
        "Once upon a time",
        "The capital of France is",
        "In machine learning,",
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=cfg["max_new_tokens"])
    start = time.perf_counter()
    outs = engine.generate_batch(prompts, sp)
    elapsed = time.perf_counter() - start
    for p, o in zip(prompts, outs):
        print(f"\n>>> {p}\n{o}")
    total_tok = sum(len(engine.tokenizer.encode(o)) for o in outs)
    print(f"\nBatch: {len(prompts)} reqs, ~{total_tok} toks in {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="vllm-v3 FlashInfer + CUDA Graph + Continuous Batching"
    )
    p.add_argument("--config", default=None, help="YAML 配置文件路径")
    p.add_argument("--model", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--cg", action="store_true", help="启用 Decode CUDA Graph")
    p.add_argument("--compile", action="store_true", dest="cg",
                   help="(别名) 启用 Decode CUDA Graph")
    p.add_argument("--prefill-cg", action="store_true", help="启用 prefill piecewise CUDA Graph")
    p.add_argument("--prefix-backend", choices=("none", "hash", "radix"), default=None)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--batch", action="store_true", help="运行批量推理演示")
    p.add_argument("--download", action="store_true")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    # 管理命令
    if args.list:
        vram = gpu_memory_gb() or 24.0
        selected = select_models_for_vram(vram_gb=vram)
        print(f"Cache: {DEFAULT_CACHE_DIR}")
        print(f"VRAM:  {vram:.1f} GB")
        print(f"All:   {', '.join(s.local_name for s in QWEN3_REGISTRY)}")
        print(f"Fit:   {', '.join(s.local_name for s in selected)}")
        return
    if args.download:
        print(f"Downloading to {DEFAULT_CACHE_DIR} ...")
        download_models(list(QWEN3_REGISTRY))
        print("Done.")
        return

    config_path = Path(args.config) if args.config else HERE / "config.yaml"
    cfg = load_config(config_path)
    for key in ("model", "prompt", "max_new_tokens", "temperature", "top_p", "top_k",
                "prefix_backend", "max_num_seqs"):
        if (v := getattr(args, key)) is not None:
            cfg[key] = v
    if args.cg:
        cfg["enforce_eager"] = False
    if args.prefill_cg:
        cfg["prefill_cuda_graph"] = True

    if args.batch:
        run_batch(cfg)
    else:
        run_single(cfg)


if __name__ == "__main__":
    main()
