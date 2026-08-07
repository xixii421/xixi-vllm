"""vllm-v0 执行入口
用法:
    python run.py                                       # 使用 config.yaml 默认值
    python run.py --model Qwen3-1.5B --prompt "你好"    # CLI 覆盖配置
    python run.py --config my_config.yaml               # 指定配置文件
    python run.py --list                                # 列出所有可用模型
    
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

from src.config import EngineConfig
from src.engine import Engine
from src.model_hub import (
    DEFAULT_CACHE_DIR,
    QWEN3_REGISTRY,
    download_models,
    gpu_memory_gb,
    select_models_for_vram,
)

# 配置加载
HERE = Path(__file__).resolve().parent
DEFAULTS = {
    "model": "Qwen3-0.6B",
    "prompt": "Hello, my name is",
    "max_new_tokens": 200,
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 20,
    "dtype": "auto",
}


def load_config(path: Path) -> dict:
    """加载 YAML 配置文件"""
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    return {k: cfg.get(k, v) for k, v in DEFAULTS.items()}


# 核心逻辑
def run(cfg: dict) -> None:
    # 模型固定存放在 ~/huggingface/<模型名>
    model_path = DEFAULT_CACHE_DIR / cfg["model"]
    print(f"Loading {model_path} ...")
    engine = Engine(
        EngineConfig(
            model=str(model_path),
            dtype=cfg["dtype"],
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





def argparse_cli() -> None:
    p = argparse.ArgumentParser(description="vllm-v0 基础推理")
    p.add_argument("--config", default=None, help="YAML 配置文件路径")
    p.add_argument("--model", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--download", action="store_true")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    # 管理命令（不需要加载模型）
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
        download_models()
        print("Done.")
        return

    # 加载配置并合并 CLI 参数
    config_path = Path(args.config) if args.config else HERE / "config.yaml"
    cfg = load_config(config_path)
    for key in ("model", "prompt", "max_new_tokens", "temperature", "top_p", "top_k"):
        if (v := getattr(args, key)) is not None:
            cfg[key] = v

    return cfg


if __name__ == "__main__":
    cfg = argparse_cli()
    run(cfg)
    
