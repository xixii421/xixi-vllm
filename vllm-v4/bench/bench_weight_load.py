#!/usr/bin/env python3
"""Benchmark model weight loading speed (disk→CPU→GPU transfer) for v4 / v5 / official.

Times the full "create model + load weights" process.
Reports both total time and weight-only time.

Usage:
  PYTHONPATH=vllm-v4  python bench/bench_weight_load.py --version v4 --model <path>
  PYTHONPATH=vllm-v5  python bench/bench_weight_load.py --version v5 --model <path>
  python bench/bench_weight_load.py --version official --model <path>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch


def _disk_size_gb(model_path: str) -> float:
    total = 0
    for f in Path(model_path).rglob("*.safetensors"):
        total += f.stat().st_size
    return total / (1024**3)


def _peak_gpu_gb() -> float:
    return torch.cuda.max_memory_allocated() / (1024**3)


def _safetensors_count(model_path: str) -> int:
    return len(list(Path(model_path).rglob("*.safetensors")))


# ── v4 ──────────────────────────────────────────────────────────────────


def run_v4(args) -> dict:
    from src.models.loader import load_model
    from src.models.qwen3 import Qwen3ForCausalLM
    from src.models.qwen3_vl import Qwen3VLForCausalLM
    from transformers import AutoConfig

    hf_cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    is_vl = bool(getattr(hf_cfg, "vision_config", None)) or "VL" in type(hf_cfg).__name__
    dev = torch.device("cuda:0")

    times_total = []
    times_weight = []
    mems = []
    for trial in range(args.iters):
        torch.cuda.reset_peak_memory_stats()

        # Phase A: create model on GPU + dtype cast
        t0 = time.perf_counter()
        if is_vl:
            model = Qwen3VLForCausalLM(hf_cfg).to(dtype=torch.float16, device=dev).eval()
        else:
            model = Qwen3ForCausalLM(hf_cfg).to(dtype=torch.float16, device=dev).eval()
        torch.cuda.synchronize()
        t_after_create = time.perf_counter()

        if is_vl:
            model.load_weights(args.model, dev, torch.float16)
        else:
            load_model(model, args.model)
        torch.cuda.synchronize()
        elapsed_total = time.perf_counter() - t0
        elapsed_weight = time.perf_counter() - t_after_create

        mem = _peak_gpu_gb()
        times_total.append(elapsed_total)
        times_weight.append(elapsed_weight)
        mems.append(mem)
        del model
        torch.cuda.empty_cache()
        print(f"  trial {trial}: total={elapsed_total:.3f}s  "
              f"(create={elapsed_total-elapsed_weight:.3f}s + load={elapsed_weight:.3f}s)  "
              f"peak_gpu={mem:.2f} GiB")

    return _report("v4", args.model, is_vl, times_total, times_weight, mems)


# ── v5 ──────────────────────────────────────────────────────────────────


def _init_v5_dist():
    """Minimal single-process distributed init matching v5's ensure_single_process_parallel."""
    from src.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        is_initialized,
    )

    if is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    init_distributed_environment(
        world_size=1, rank=0, backend="nccl",
        init_method="tcp://127.0.0.1:29501", local_device=0,
    )
    initialize_model_parallel(tp_size=1, pp_size=1, tp_rank=0, pp_rank=0)


def run_v5(args) -> dict:
    from src.config.engine import _is_vl_config
    from transformers import AutoConfig

    _init_v5_dist()
    hf_cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    is_vl = _is_vl_config(hf_cfg)
    dev = torch.device("cuda:0")

    if is_vl:
        from src.models.qwen3_vl import Qwen3VLForCausalLM
        def _make(): return Qwen3VLForCausalLM(hf_cfg, mm_encoder_tp_mode="data")
    else:
        from src.models.qwen3 import Qwen3ForCausalLM
        def _make(): return Qwen3ForCausalLM(hf_cfg)

    times_total = []
    times_weight = []
    mems = []
    for trial in range(args.iters):
        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        model = _make().to(dtype=torch.float16, device=dev).eval()
        torch.cuda.synchronize()
        t_after_create = time.perf_counter()

        if is_vl:
            model.load_weights(args.model, dev, torch.float16, rank=0, tp_size=1, pp_size=1)
        else:
            from src.models.loader import load_model_hf
            load_model_hf(model, args.model)
        torch.cuda.synchronize()
        elapsed_total = time.perf_counter() - t0
        elapsed_weight = time.perf_counter() - t_after_create

        mem = _peak_gpu_gb()
        times_total.append(elapsed_total)
        times_weight.append(elapsed_weight)
        mems.append(mem)
        del model
        torch.cuda.empty_cache()
        print(f"  trial {trial}: total={elapsed_total:.3f}s  "
              f"(create={elapsed_total-elapsed_weight:.3f}s + load={elapsed_weight:.3f}s)  "
              f"peak_gpu={mem:.2f} GiB")

    return _report("v5", args.model, is_vl, times_total, times_weight, mems)


# ── Official vLLM (HF transformers) ───────────────────────────────────


def run_official(args) -> dict:
    """Use HF transformers (same underlying loader as vLLM)."""
    from transformers import AutoConfig, AutoModel

    hf_cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    is_vl = bool(getattr(hf_cfg, "vision_config", None)) or "VL" in type(hf_cfg).__name__

    times_total = []
    times_weight = []
    mems = []
    for trial in range(args.iters):
        torch.cuda.reset_peak_memory_stats()

        # Phase A: load from disk → CPU memory (HF does both creation + I/O here)
        t0 = time.perf_counter()
        model = AutoModel.from_pretrained(
            args.model, dtype=torch.float16, trust_remote_code=True,
            attn_implementation=None, device_map=None,
        )
        t_after_io = time.perf_counter()

        # Phase B: copy to GPU
        model = model.to("cuda")
        torch.cuda.synchronize()
        elapsed_total = time.perf_counter() - t0
        elapsed_weight = time.perf_counter() - t_after_io

        mem = _peak_gpu_gb()
        times_total.append(elapsed_total)
        times_weight.append(elapsed_weight)
        mems.append(mem)
        del model
        torch.cuda.empty_cache()
        print(f"  trial {trial}: total={elapsed_total:.3f}s  "
              f"(disk→CPU={elapsed_total-elapsed_weight:.3f}s + CPU→GPU={elapsed_weight:.3f}s)  "
              f"peak_gpu={mem:.2f} GiB")

    return _report("official", args.model, is_vl, times_total, times_weight, mems)


# ── helpers ─────────────────────────────────────────────────────────────


def _report(backend, model_path, is_vl, times_total, times_weight, mems):
    n = len(times_total)
    def _s(x): return statistics.stdev(x) if len(x) >= 2 else 0.0
    return {
        "backend": backend,
        "model": model_path,
        "is_multimodal": is_vl,
        "disk_size_gb": round(_disk_size_gb(model_path), 3),
        "n_safetensors": _safetensors_count(model_path),
        "device": torch.cuda.get_device_name(0),
        "iters": n,
        "total_s": [round(t, 4) for t in times_total],
        "load_s":   [round(w, 4) for w in times_weight],
        "create_s": [round(t - w, 4) for t, w in zip(times_total, times_weight)],
        "peak_gpu_gb": [round(m, 2) for m in mems],
        "total_mean_s":   round(statistics.mean(times_total), 4),
        "total_median_s": round(statistics.median(times_total), 4),
        "total_stdev_s":  round(_s(times_total), 4),
        "total_min_s":    round(min(times_total), 4),
        "total_max_s":    round(max(times_total), 4),
        "weight_mean_s":   round(statistics.mean(times_weight), 4),
        "weight_median_s": round(statistics.median(times_weight), 4),
        "weight_stdev_s":  round(_s(times_weight), 4),
        "create_mean_s":   round(statistics.mean([t - w for t, w in zip(times_total, times_weight)]), 4),
        "create_median_s": round(statistics.median([t - w for t, w in zip(times_total, times_weight)]), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["v4", "v5", "official"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"model={args.model}")
    print(f"  safe_files={_safetensors_count(args.model)}")
    print(f"  disk_size={_disk_size_gb(args.model):.3f} GiB")
    print(f"version={args.version} iters={args.iters}")
    print()

    if args.version == "v4":
        report = run_v4(args)
    elif args.version == "v5":
        report = run_v5(args)
    else:
        report = run_official(args)

    print()
    print(f"==== {args.version} @ {args.model} ====")
    t = report
    print(f"  total:   mean={t['total_mean_s']:.3f}s median={t['total_median_s']:.3f}s "
          f"stdev={t['total_stdev_s']:.3f}s [{t['total_min_s']:.3f}–{t['total_max_s']:.3f}]")
    print(f"  └ create:{t['create_mean_s']:.3f}s median={t['create_median_s']:.3f}s")
    print(f"  └ load:  {t['weight_mean_s']:.3f}s median={t['weight_median_s']:.3f}s")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"wrote {out}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
