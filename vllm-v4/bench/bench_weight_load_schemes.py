#!/usr/bin/env python3
"""Compare safetensors load strategies end-to-end (read + write into CUDA params).

Usage::

    cd vllm-v4
    source ../.venv/bin/activate && source ../.env.cuda.sh
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python bench/bench_weight_load_schemes.py \\
        --model /mnt/nfs/zhangli/huggingface/Qwen3-0.6B
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python bench/bench_weight_load_schemes.py \\
        --model /mnt/nfs/zhangli/huggingface/Qwen3-VL-8B-Instruct --repeats 2
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import os
import statistics
import time
from collections.abc import Callable, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def _files(hf_folder: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(hf_folder, "*.safetensors")))
    if not files:
        raise FileNotFoundError(hf_folder)
    return files


def _load_shard_cpu_safe_open(path: str) -> list[tuple[str, torch.Tensor]]:
    with safe_open(path, framework="pt", device="cpu") as f:
        return [(n, f.get_tensor(n)) for n in f.keys()]


def _load_shard_cpu_load_file(path: str) -> list[tuple[str, torch.Tensor]]:
    sd = load_file(path, device="cpu")
    return list(sd.items())


# ----- iterators (yield name, tensor) -----


def scheme_A_safe_open_cpu(hf_folder: str, device: str) -> Iterable[tuple[str, torch.Tensor]]:
    for path in _files(hf_folder):
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def scheme_B_load_file_cpu(hf_folder: str, device: str) -> Iterable[tuple[str, torch.Tensor]]:
    for path in _files(hf_folder):
        sd = load_file(path, device="cpu")
        yield from sd.items()
        del sd


def scheme_C_load_file_cuda(hf_folder: str, device: str) -> Iterable[tuple[str, torch.Tensor]]:
    for path in _files(hf_folder):
        sd = load_file(path, device=device)
        yield from sd.items()
        del sd


def scheme_D_pipeline_safe_open(
    hf_folder: str, device: str, max_workers: int = 2
) -> Iterable[tuple[str, torch.Tensor]]:
    files = _files(hf_folder)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_load_shard_cpu_safe_open, p) for p in files]
        for fut in futures:
            for name, tensor in fut.result():
                yield name, tensor.to(device, non_blocking=True)


def scheme_E_pipeline_load_file(
    hf_folder: str, device: str, max_workers: int = 2
) -> Iterable[tuple[str, torch.Tensor]]:
    """Prefetch shards with load_file(cpu); H2D with non_blocking."""
    files = _files(hf_folder)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_load_shard_cpu_load_file, p) for p in files]
        for fut in futures:
            for name, tensor in fut.result():
                yield name, tensor.to(device, non_blocking=True)


def scheme_F_safe_open_direct_gpu(hf_folder: str, device: str) -> Iterable[tuple[str, torch.Tensor]]:
    for path in _files(hf_folder):
        with safe_open(path, framework="pt", device=device) as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def scheme_G_prefetch1_load_file(
    hf_folder: str, device: str
) -> Iterable[tuple[str, torch.Tensor]]:
    """One-ahead prefetch: load next shard on CPU while yielding current (still on CPU)."""
    files = _files(hf_folder)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_load_shard_cpu_load_file, files[0])
        for i, _path in enumerate(files):
            items = fut.result()
            if i + 1 < len(files):
                fut = ex.submit(_load_shard_cpu_load_file, files[i + 1])
            yield from items


def scheme_H_prefetch1_to_gpu(
    hf_folder: str, device: str
) -> Iterable[tuple[str, torch.Tensor]]:
    """Prefetch shard on CPU; yield GPU tensors via non_blocking .to (caller should D2D once)."""
    files = _files(hf_folder)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_load_shard_cpu_load_file, files[0])
        for i, _path in enumerate(files):
            items = fut.result()
            if i + 1 < len(files):
                fut = ex.submit(_load_shard_cpu_load_file, files[i + 1])
            for name, tensor in items:
                yield name, tensor.to(device, non_blocking=True)


SCHEMES: dict[str, Callable[..., Iterable[tuple[str, torch.Tensor]]]] = {
    "A_safe_open_cpu": scheme_A_safe_open_cpu,
    "B_load_file_cpu": scheme_B_load_file_cpu,
    "C_load_file_cuda": scheme_C_load_file_cuda,
    "D_pipe_safe_open": scheme_D_pipeline_safe_open,
    "E_pipe_load_file": scheme_E_pipeline_load_file,
    "F_safe_open_gpu": scheme_F_safe_open_direct_gpu,
    "G_prefetch_cpu_yield": scheme_G_prefetch1_load_file,
    "H_prefetch_to_gpu": scheme_H_prefetch1_to_gpu,
}


def run_into_params(
    it: Iterable[tuple[str, torch.Tensor]],
    device: str,
    *,
    non_blocking_h2d: bool,
) -> tuple[int, int]:
    """Simulate weight_loader: copy each tensor into a CUDA parameter buffer."""
    n = 0
    nbytes = 0
    hold: list[torch.Tensor] = []
    for _name, w in it:
        n += 1
        nbytes += w.numel() * w.element_size()
        param = torch.empty(w.shape, dtype=w.dtype, device=device)
        if w.device.type == "cuda":
            param.copy_(w, non_blocking=True)
        else:
            param.copy_(w, non_blocking=non_blocking_h2d)
        hold.append(param)
        del w
    torch.cuda.synchronize()
    del hold
    return n, nbytes


def bench_scheme(
    name: str,
    fn: Callable,
    hf_folder: str,
    device: str,
    repeats: int,
    warmup: int,
    *,
    env_fast_gpu: bool = False,
) -> float:
    prev = os.environ.get("SAFETENSORS_FAST_GPU")
    if env_fast_gpu:
        os.environ["SAFETENSORS_FAST_GPU"] = "1"
    elif "SAFETENSORS_FAST_GPU" in os.environ:
        del os.environ["SAFETENSORS_FAST_GPU"]

    # CPU-yielding schemes: use non_blocking H2D into params (sync at end).
    nb = not name.startswith("C_") and not name.startswith("F_")

    def once() -> None:
        run_into_params(fn(hf_folder, device), device, non_blocking_h2d=nb)

    try:
        for _ in range(warmup):
            once()
            torch.cuda.empty_cache()
        times = []
        for _ in range(repeats):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            once()
            times.append(time.perf_counter() - t0)
            torch.cuda.empty_cache()
        return statistics.median(times)
    finally:
        if prev is None:
            os.environ.pop("SAFETENSORS_FAST_GPU", None)
        else:
            os.environ["SAFETENSORS_FAST_GPU"] = prev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/nfs/zhangli/huggingface/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    print(f"model={args.model}")
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"shards={len(_files(args.model))}")
    print(f"{'scheme':<28} {'median_ms':>10} {'vs_best':>10}")
    print("-" * 52)

    results: dict[str, float] = {}

    # Baseline + main schemes
    for name, fn in SCHEMES.items():
        try:
            med = bench_scheme(
                name, fn, args.model, args.device, args.repeats, args.warmup
            )
            results[name] = med
            print(f"{name:<28} {med*1e3:10.1f}")
        except Exception as e:
            print(f"{name:<28} FAILED: {type(e).__name__}: {e}")

    # F with SAFETENSORS_FAST_GPU=1
    try:
        med = bench_scheme(
            "F_safe_open_gpu",
            scheme_F_safe_open_direct_gpu,
            args.model,
            args.device,
            args.repeats,
            args.warmup,
            env_fast_gpu=True,
        )
        results["F_safe_open_gpu+FAST_GPU"] = med
        print(f"{'F_safe_open_gpu+FAST_GPU':<28} {med*1e3:10.1f}")
    except Exception as e:
        print(f"{'F_safe_open_gpu+FAST_GPU':<28} FAILED: {e}")

    # C with FAST_GPU
    try:
        med = bench_scheme(
            "C_load_file_cuda",
            scheme_C_load_file_cuda,
            args.model,
            args.device,
            args.repeats,
            args.warmup,
            env_fast_gpu=True,
        )
        results["C_load_file_cuda+FAST_GPU"] = med
        print(f"{'C_load_file_cuda+FAST_GPU':<28} {med*1e3:10.1f}")
    except Exception as e:
        print(f"{'C_load_file_cuda+FAST_GPU':<28} FAILED: {e}")

    best_name = min(results, key=results.get)
    best = results[best_name]
    print("-" * 52)
    print(f"BEST: {best_name}  {best*1e3:.1f} ms")
    print()
    for name, t in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {name:<28} {t*1e3:8.1f} ms   {t/best:5.2f}x of best")


if __name__ == "__main__":
    main()
