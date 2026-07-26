#!/usr/bin/env python3
"""A/B microbench: safetensors iterator (old safe_open vs new load_file).

Usage::

    cd vllm-v4
    source ../.venv/bin/activate && source ../.env.cuda.sh
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python bench/bench_weight_load.py \\
        --model /mnt/nfs/zhangli/huggingface/Qwen3-0.6B
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python bench/bench_weight_load.py \\
        --model /mnt/nfs/zhangli/huggingface/Qwen3-VL-8B-Instruct --full-model
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def iter_old(hf_folder: str) -> Iterable[tuple[str, torch.Tensor]]:
    """Previous loader.py path: per-tensor safe_open on CPU."""
    files = sorted(glob.glob(os.path.join(hf_folder, "*.safetensors")))
    if not files:
        raise FileNotFoundError(hf_folder)
    for st_file in files:
        with safe_open(st_file, framework="pt") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def iter_new(
    hf_folder: str, device: str = "cuda"
) -> Iterable[tuple[str, torch.Tensor]]:
    """Current loader.py path: whole-file load_file onto device."""
    files = sorted(glob.glob(os.path.join(hf_folder, "*.safetensors")))
    if not files:
        raise FileNotFoundError(hf_folder)
    for st_file in files:
        state_dict = load_file(st_file, device=device)
        for name, tensor in state_dict.items():
            yield name, tensor
        del state_dict


def _drain(it: Iterable[tuple[str, torch.Tensor]], *, to_device: str | None) -> tuple[int, int]:
    """Materialize iterator; optionally copy each tensor to ``to_device`` (simulates param.copy_)."""
    n = 0
    nbytes = 0
    for _name, t in it:
        n += 1
        nbytes += t.numel() * t.element_size()
        if to_device is not None:
            # Mimic default_weight_loader onto a same-shaped buffer on target device.
            dst = torch.empty_like(t, device=to_device)
            dst.copy_(t)
            del dst
        del t
    if to_device is not None and to_device.startswith("cuda"):
        torch.cuda.synchronize()
    return n, nbytes


def bench_iter(
    label: str,
    fn,
    *,
    repeats: int,
    warmup: int,
    to_device: str | None,
) -> list[float]:
    for _ in range(warmup):
        _drain(fn(), to_device=to_device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    times = []
    last_n = last_b = 0
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        t0 = time.perf_counter()
        last_n, last_b = _drain(fn(), to_device=to_device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(
        f"  {label:<28} median={statistics.median(times)*1e3:8.1f} ms  "
        f"min={min(times)*1e3:8.1f} ms  tensors={last_n}  "
        f"payload={last_b/1e9:.2f} GB"
    )
    return times


def bench_full_model(model_path: str, device: torch.device) -> None:
    """End-to-end Engine-style load (VL or text) with current loader."""
    from transformers import AutoConfig

    from src.config import EngineConfig, is_multimodal
    from src.models.loader import load_model, safetensors_weights_iterator
    from src.models.qwen3 import Qwen3ForCausalLM
    from src.models.qwen3_vl import Qwen3VLForCausalLM

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"\n[full-model] {model_path} on {device}")

    def once_new() -> float:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if is_multimodal(cfg):
            model = Qwen3VLForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
            model.load_weights(model_path, device, torch.bfloat16)
        else:
            model = Qwen3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
            load_model(model, model_path)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        del model
        torch.cuda.empty_cache()
        return dt

    # Old path: force CPU safe_open style by temporarily patching
    import src.models.loader as loader_mod
    import src.models.qwen3_vl as vl_mod

    def once_old() -> float:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        def _old_iter(hf_folder, device=None):  # noqa: ARG001
            yield from iter_old(hf_folder)

        orig = loader_mod.safetensors_weights_iterator
        loader_mod.safetensors_weights_iterator = _old_iter  # type: ignore[assignment]
        vl_mod.safetensors_weights_iterator = _old_iter  # type: ignore[assignment]
        try:
            t0 = time.perf_counter()
            if is_multimodal(cfg):
                model = Qwen3VLForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
                model.load_weights(model_path, device, torch.bfloat16)
            else:
                model = Qwen3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
                # text path uses load_model → iterator without device kw
                model.load_weights(iter_old(model_path))
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            del model
        finally:
            loader_mod.safetensors_weights_iterator = orig
            vl_mod.safetensors_weights_iterator = orig
        torch.cuda.empty_cache()
        return dt

    # warmup construct once lightly skipped — full load is heavy
    t_old = once_old()
    t_new = once_new()
    print(f"  old (safe_open CPU→copy)     {t_old*1e3:8.1f} ms")
    print(f"  new (load_file→device)       {t_new*1e3:8.1f} ms")
    if t_new > 0:
        print(f"  speedup                      {t_old/t_new:8.2f}x")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="/mnt/nfs/zhangli/huggingface/Qwen3-0.6B",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument(
        "--full-model",
        action="store_true",
        help="Also time constructing + load_weights end-to-end",
    )
    args = ap.parse_args()

    if not Path(args.model).is_dir():
        raise SystemExit(f"model not found: {args.model}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this bench")

    device = args.device
    print(f"model={args.model}")
    print(f"cuda={torch.cuda.get_device_name(0)} (visible device 0)")
    print(f"files={len(sorted(glob.glob(os.path.join(args.model, '*.safetensors'))))}")

    print("\n[1] Iterator only (no H2D copy) — pure read path")
    t_old = bench_iter(
        "old safe_open(CPU)",
        lambda: iter_old(args.model),
        repeats=args.repeats,
        warmup=args.warmup,
        to_device=None,
    )
    t_new_cpu = bench_iter(
        "new load_file(cpu)",
        lambda: iter_new(args.model, device="cpu"),
        repeats=args.repeats,
        warmup=args.warmup,
        to_device=None,
    )
    t_new_gpu = bench_iter(
        "new load_file(cuda)",
        lambda: iter_new(args.model, device=device),
        repeats=args.repeats,
        warmup=args.warmup,
        to_device=None,
    )

    print("\n[2] Iterator + copy_ onto CUDA (simulates weight_loader)")
    t_old_h2d = bench_iter(
        "old CPU + copy_→cuda",
        lambda: iter_old(args.model),
        repeats=args.repeats,
        warmup=args.warmup,
        to_device=device,
    )
    t_new_h2d = bench_iter(
        "new cuda + copy_→cuda",
        lambda: iter_new(args.model, device=device),
        repeats=args.repeats,
        warmup=args.warmup,
        to_device=device,
    )

    def ratio(a, b):
        return statistics.median(a) / statistics.median(b)

    print("\n[summary]")
    print(f"  load_file(cpu)  vs safe_open : {ratio(t_old, t_new_cpu):.2f}x")
    print(f"  load_file(cuda) vs safe_open : {ratio(t_old, t_new_gpu):.2f}x  (read path)")
    print(f"  new+copy vs old+copy         : {ratio(t_old_h2d, t_new_h2d):.2f}x")

    if args.full_model:
        bench_full_model(args.model, torch.device(device))


if __name__ == "__main__":
    main()
