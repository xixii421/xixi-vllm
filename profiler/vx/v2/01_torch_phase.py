"""剖析 v2 单请求的 prefill / decode 阶段（torch.profiler）。

单请求天然分离两阶段：第 1 步是 Prefill，之后每步是 Decode。
每步打 record_function 标签，输出 CPU/GPU 对照表 + Chrome trace。
Perfetto: https://ui.perfetto.dev/
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "vllm-v2"))

from src.config import EngineConfig
from src.engine import Engine
from src.model_hub import DEFAULT_CACHE_DIR
from src.sampling_params import SamplingParams


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-len", type=int, default=256, help="prompt token 数")
    p.add_argument("--max-new-tokens", type=int, default=32, help="decode 步数")
    p.add_argument("--warm", type=int, default=8, help="剖析前预热 decode 步数")
    p.add_argument("--row-limit", type=int, default=20, help="算子表行数")
    p.add_argument("--trace_dir", default="./out", help="trace 输出目录")
    return p.parse_args()


def main():
    args = parse_arguments()

    model = DEFAULT_CACHE_DIR / "Qwen3-0.6B"
    engine = Engine(
        EngineConfig(
            model=str(model),
            context_len=2048,
            block_size=16,
            prefix_backend="hash",
            max_num_seqs=8,
            max_num_batched_tokens=4096,
            mix_prefill_decode=True,
        )
    )
    base = engine.tokenizer.encode("The quick brown fox jumps over the lazy dog.")
    tokens = (base * (args.prompt_len // len(base) + 1))[: args.prompt_len]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    # 预热轮不剖析：GPU 分配/缓存先就位，避免污染数据
    for _ in range(args.warm):
        seq = engine.add_request(tokens, sp)
        while not seq.is_finished:
            engine.step()

    # 正式轮：每步按 running 队列区分 prefill/decode 并打标签
    schedule = torch.profiler.schedule(wait=0, warmup=0,
                                       active=args.max_new_tokens + 1, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule,
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        seq = engine.add_request(tokens, sp)
        with record_function("infer_loop"):
            while not seq.is_finished:
                phase = "prefill" if not engine.scheduler.running else "decode"
                with record_function(phase):
                    engine.step()
                prof.step()

    torch.cuda.synchronize()

    # 输出：阶段汇总 + 算子级 CPU/GPU 对照表
    print("=== 阶段汇总 (CPU 启动 vs GPU 执行) ===")
    for phase in ("prefill", "decode"):
        evs = [e for e in prof.events() if e.name == phase]
        if not evs:
            continue
        cpu_ms = sum(e.cpu_time_total for e in evs) / 1000
        gpu_ms = sum(e.device_time_total for e in evs) / 1000
        print(f"  {phase:7s} CPU {cpu_ms:8.1f}ms   GPU {gpu_ms:8.1f}ms   "
              f"CPU/GPU {cpu_ms/gpu_ms:5.2f}x")
    print("\n" + prof.key_averages().table(sort_by="device_time_total",
                                           row_limit=args.row_limit))

    # 落盘：算子表 + Chrome trace
    os.makedirs(args.trace_dir, exist_ok=True)
    with open(f"{args.trace_dir}/torch_profile.txt", "w") as f:
        f.write(prof.key_averages().table(sort_by="device_time_total", row_limit=100))
    prof.export_chrome_trace(f"{args.trace_dir}/torch_profile_trace.json")
    print(f"Chrome trace → {args.trace_dir}/torch_profile_trace.json")


if __name__ == "__main__":
    main()
