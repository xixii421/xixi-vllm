"""vllm-v0 统一入口 — 基础自回归推理。

用法:
    python run.py                                    # 使用 config.yaml 默认值
    python run.py --model Qwen3-1.5B --prompt "你好"   # CLI 覆盖配置
    python run.py --config my_config.yaml             # 指定配置文件
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.run_utils import handle_hub_cli, load_config, merge_cli  # noqa: E402

from src.config import EngineConfig  # noqa: E402
from src.engine import Engine  # noqa: E402
from src.model_hub import resolve_model_path  # noqa: E402

DEFAULTS = {
    "model": "Qwen3-0.6B",
    "prompt": "Hello, my name is",
    "max_new_tokens": 200,
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 20,
    "enforce_eager": True,
    "dtype": "auto",
}


def run(cfg: dict) -> None:
    model_path = resolve_model_path(cfg["model"])
    print(f"Loading {model_path} ...")
    engine = Engine(
        EngineConfig(
            model=str(model_path),
            enforce_eager=cfg["enforce_eager"],
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


def main() -> None:
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

    if handle_hub_cli(args):
        return

    config_path = Path(args.config) if args.config else HERE / "config.yaml"
    cfg = load_config(config_path, DEFAULTS)
    merge_cli(cfg, args, ("model", "prompt", "max_new_tokens", "temperature", "top_p", "top_k"))
    run(cfg)


if __name__ == "__main__":
    main()
