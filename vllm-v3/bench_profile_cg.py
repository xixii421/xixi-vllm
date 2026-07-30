"""Profile vllm-v3 — CUDA Graph mode (FlashInfer + Decode CG).

Phase annotations:
  engine.step → schedule, model_runner, postprocess

Decode path automatically uses CUDA Graph when batch size matches captured sizes.
Prefill is always eager (FlashInfer paged prefill kernel).

Usage:
    python bench_profile_cg.py                    # CUDA Graph profiling
    python bench_profile_cg.py --compare-eager    # Also run eager for comparison

Output: Chrome-trace JSON + op-level table + category summary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from src.config import EngineConfig
from src.engine.engine import Engine
from src.sampling_params import SamplingParams
from profiler import Profiler


def run_profile_cg(model_path: str, max_new: int = 32) -> float:
    """Run CUDA Graph profile. Returns total wall-clock time (s)."""
    cfg = EngineConfig(
        model=model_path,
        enforce_eager=False,  # enable CUDA Graph decode
        context_len=512,
        prefix_backend="hash",
        max_num_seqs=8,
        max_num_batched_tokens=2048,
    )
    t0 = time.perf_counter()
    engine = Engine(cfg)  # CUDA Graph capture happens here
    t_capture = time.perf_counter() - t0
    print(f"CG capture time: {t_capture:.2f}s")

    prompts = ["The capital of France is", "Hello, my name is", "Once upon a time"]
    sp = SamplingParams(temperature=0.0, max_tokens=max_new)
    for p in prompts:
        engine.add_request(p, sp)

    with Profiler(trace_dir="./traces/v3", label="v3_cg") as prof:
        step_count = 0
        while not engine.scheduler.is_finished():
            with prof.annotate("engine.step"):
                with prof.annotate("schedule"):
                    prefills, decodes = engine.scheduler.schedule()
                if not prefills and not decodes:
                    break
                with prof.annotate("model_runner"):
                    p_ids, d_ids = engine.model_runner.run(prefills, decodes)
                with prof.annotate("postprocess"):
                    if decodes:
                        engine.scheduler.postprocess(decodes, d_ids, is_prefill=False)
                    if prefills:
                        engine.scheduler.postprocess(prefills, p_ids, is_prefill=True)
            step_count += 1

    engine.destroy()
    elapsed = time.perf_counter() - t0

    prof.print_stats()
    prof.print_category_summary()
    print(f"\n[profiler] {step_count} steps, traces → {prof.trace_dir}/v3_cg*.json")
    print(f"           mode:  CUDA GRAPH (decode), FlashInfer (prefill)")
    print(f"           wall:  {elapsed:.2f}s (incl. {t_capture:.2f}s CG capture)")
    return elapsed


def run_profile_eager(model_path: str, max_new: int = 32) -> float:
    """Run eager mode for comparison. Returns total wall-clock time (s)."""
    cfg = EngineConfig(
        model=model_path,
        enforce_eager=True,
        context_len=512,
        prefix_backend="hash",
        max_num_seqs=8,
        max_num_batched_tokens=2048,
    )
    engine = Engine(cfg)

    prompts = ["The capital of France is", "Hello, my name is", "Once upon a time"]
    sp = SamplingParams(temperature=0.0, max_tokens=max_new)
    for p in prompts:
        engine.add_request(p, sp)

    t0 = time.perf_counter()
    step_count = 0
    while not engine.scheduler.is_finished():
        prefills, decodes = engine.scheduler.schedule()
        if not prefills and not decodes:
            break
        engine.model_runner.run(prefills, decodes)
        if decodes:
            engine.scheduler.postprocess(decodes, [], is_prefill=False)
        if prefills:
            engine.scheduler.postprocess(prefills, [], is_prefill=True)
        step_count += 1

    engine.destroy()
    elapsed = time.perf_counter() - t0
    print(f"[compare] eager: {step_count} steps, {elapsed:.3f}s")
    return elapsed


def main():
    p = argparse.ArgumentParser(description="Profile vllm-v3 (CUDA Graph)")
    p.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--compare-eager", action="store_true",
                   help="Also run eager mode for comparison")
    args = p.parse_args()

    t_cg = run_profile_cg(args.model, args.max_new)
    if args.compare_eager:
        t_eager = run_profile_eager(args.model, args.max_new)
        print(f"\n=== Comparison ===")
        print(f"  CUDA Graph:  {t_cg:.3f}s")
        print(f"  Eager:       {t_eager:.3f}s")
        print(f"  Speedup:     {t_eager / t_cg:.2f}x")


if __name__ == "__main__":
    main()
