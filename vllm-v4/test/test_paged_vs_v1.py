"""Greedy consistency: v2 PagedAttention vs v1 contiguous KV Cache.

Usage:
  ../vllm-v0/.venv/bin/python test/test_paged_vs_v1.py --model Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
V1 = REPO / "vllm-v1"
V2 = ROOT


def _purge_src() -> None:
    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            del sys.modules[key]


def _load_engine(version_dir: Path, model_path: str, context_len: int):
    import importlib

    _purge_src()
    if str(version_dir) in sys.path:
        sys.path.remove(str(version_dir))
    sys.path.insert(0, str(version_dir))
    EngineConfig = importlib.import_module("src.config").EngineConfig
    Engine = importlib.import_module("src.engine").Engine
    kwargs = {
        "model": model_path,
        "context_len": context_len,
        "enforce_eager": True,
    }
    if version_dir.name == "vllm-v2":
        kwargs.update(block_size=16, max_num_seqs=2)
    engine = Engine(EngineConfig(**kwargs))
    return engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen3-0.6B")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--prompt", default="Hello, my name is")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        model_path = Path.home() / "huggingface" / args.model
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    print(f"Loading v1 from {V1} ...")
    eng_v1 = _load_engine(V1, str(model_path), args.context_len)
    out_v1 = list(
        eng_v1.generate(args.prompt, max_new_tokens=args.max_new_tokens, temperature=0.0)
    )
    text_v1 = "".join(out_v1)
    del eng_v1
    _purge_src()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Loading v2 from {V2} ...")
    eng_v2 = _load_engine(V2, str(model_path), args.context_len)
    out_v2 = list(
        eng_v2.generate(args.prompt, max_new_tokens=args.max_new_tokens, temperature=0.0)
    )
    text_v2 = "".join(out_v2)

    print(f"v1: {args.prompt}{text_v1}")
    print(f"v2: {args.prompt}{text_v2}")
    if text_v1 == text_v2:
        print(f"\n✅ [PASS] greedy text match ({len(out_v1)} tokens)")
        sys.exit(0)

    n = min(len(out_v1), len(out_v2))
    mismatch = next((i for i in range(n) if out_v1[i] != out_v2[i]), None)
    if mismatch is not None:
        print(
            f"\n❌ [FAIL] mismatch at piece {mismatch}: "
            f"v1={out_v1[mismatch]!r} v2={out_v2[mismatch]!r}"
        )
    else:
        print(f"\n❌ [FAIL] length {len(out_v1)} vs {len(out_v2)}")
    sys.exit(1)


if __name__ == "__main__":
    main()
