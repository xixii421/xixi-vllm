"""v2 / v3 端到端吞吐实测（与 nano-vllm / 官方 vllm 对比用同一份数据）。

读 out/e2e_prompts.json，跑连续批处理吞吐，输出 tokens/s。
所有请求一次性入队（FCFS + 连续批处理），ignore_eos=True 保证产出完整。

用法:
    python bench_e2e_data.py            # 先生成数据
    python bench_e2e_self.py --engine v2
    python bench_e2e_self.py --engine v3
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL = str(Path.home() / "huggingface" / "Qwen3-0.6B")


def make_engine(engine: str):
    sys.path.insert(0, str(ROOT) if engine == "v3" else str(ROOT.parent / "vllm-v2"))
    if engine == "v3":
        from src import Engine, EngineConfig
        from src.sampling_params import SamplingParams

        cfg = EngineConfig(
            model=MODEL,
            enforce_eager=False,
            prefill_cuda_graph=False,
            dtype="auto",
            context_len=4096,
            max_num_seqs=16,
            max_num_batched_tokens=8192,
            mix_prefill_decode=True,
            kvcache_block_size=256,
            prefix_backend="none",
            gpu_memory_utilization=0.45,
        )
    else:
        from src.config import EngineConfig
        from src.engine import Engine
        from src.sampling_params import SamplingParams

        cfg = EngineConfig(
            model=MODEL,
            context_len=4096,
            prefix_backend="none",
            block_size=16,
            max_num_seqs=16,
            max_num_batched_tokens=8192,
            mix_prefill_decode=True,
        )
    return Engine(cfg), SamplingParams


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=("v2", "v3"), required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.cuda.set_device(args.device)

    data = json.loads(
        (Path(__file__).resolve().parent / "out" / "e2e_prompts.json").read_text()
    )
    prompts, max_tokens = data["prompts"], data["max_tokens"]
    total_out = sum(max_tokens)

    engine, sp_cls = make_engine(args.engine)
    print(f"[info] engine={args.engine} ready")

    # warmup
    engine.add_request([7, 13, 29, 31, 47], sp_cls(temperature=0.6, ignore_eos=True, max_tokens=4))
    while not engine.scheduler.is_finished():
        engine.step()
    engine.reset_kv()

    seqs = [
        engine.add_request(p, sp_cls(temperature=0.6, ignore_eos=True, max_tokens=mt))
        for p, mt in zip(prompts, max_tokens)
    ]
    t0 = time.perf_counter()
    while not engine.scheduler.is_finished():
        engine.step()
    wall = time.perf_counter() - t0

    completed = sum(s.num_completion_tokens for s in seqs)
    throughput = completed / wall
    print(f"[result] engine={args.engine} wall={wall:.2f}s "
          f"output_tokens={completed}/{total_out} "
          f"throughput={throughput:.1f} tok/s")

    if hasattr(engine, "destroy"):
        engine.destroy()
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    out = Path(__file__).resolve().parent / "out"
    out.mkdir(parents=True, exist_ok=True)
    fname = f"e2e_{args.engine}.md"
    (out / fname).write_text(
        f"# e2e throughput — engine={args.engine}\n\n"
        f"- wall: {wall:.2f} s\n"
        f"- output tokens: {completed}/{total_out}\n"
        f"- throughput: {throughput:.1f} tok/s\n"
    )
    print(f"\n→ saved to {out / fname}")


if __name__ == "__main__":
    main()
