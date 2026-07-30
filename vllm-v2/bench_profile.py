"""Profile vllm-v2 — Continuous Batching + Scheduler + ModelRunner.

Phase annotations: schedule, model_runner, postprocess
Output: Chrome‑trace JSON + op‑level table + category summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from src.config import EngineConfig
from src.engine.engine import Engine
from src.sampling_params import SamplingParams
from profiler import Profiler


def run_profile(model_path: str, max_new: int = 32) -> None:
    cfg = EngineConfig(
        model=model_path,
        enforce_eager=True,
        context_len=512,
        prefix_backend="radix",
        max_num_seqs=8,
        max_num_batched_tokens=2048,
    )
    engine = Engine(cfg)
    prompts = ["The capital of France is", "Hello, my name is", "Once upon a time"]
    sp = SamplingParams(temperature=0.0, max_tokens=max_new)
    for p in prompts:
        engine.add_request(p, sp)

    with Profiler(trace_dir="./traces/v2", label="v2_eager") as prof:
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

    prof.print_stats()
    prof.print_category_summary()
    print(f"\n[profiler] {step_count} steps, traces → {prof.trace_dir}/v2_eager*.json")
    print("           view:  tensorboard --logdir traces/")


def main():
    p = argparse.ArgumentParser(description="Profile vllm-v2")
    p.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    p.add_argument("--max-new", type=int, default=32)
    args = p.parse_args()
    run_profile(args.model, args.max_new)


if __name__ == "__main__":
    main()
