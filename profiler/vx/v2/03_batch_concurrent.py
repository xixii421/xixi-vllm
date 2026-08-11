"""v2 单请求 vs 多请求并发 的剖析对比。

用同一 Engine、同一配置跑两轮：
  - single: 1 个请求，天然分离 prefill(1步) / decode(N步)
  - batch:  N 个请求同时加入, 观察 CB 合批的 prefill/decode 混合步

每步打 record_function 标签, 输出各轮 CPU/GPU 对照表 + Chrome trace。
Perfetto: https://ui.perfetto.dev/

用法:
    python 03_batch_concurrent.py --num-reqs 4 --prompt-len 128 --max-new-tokens 32
    python 03_batch_concurrent.py --mode batch   # 只跑多请求
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
    p.add_argument("--mode", choices=["both", "single", "batch"], default="both")
    p.add_argument("--num-reqs", type=int, default=4, help="多请求并发数量")
    p.add_argument("--prompt-len", type=int, default=128, help="prompt token 数")
    p.add_argument("--max-new-tokens", type=int, default=32, help="每请求生成 token 数")
    p.add_argument("--warm", type=int, default=4, help="预热轮数")
    p.add_argument("--row-limit", type=int, default=15, help="算子表行数")
    p.add_argument("--trace_dir", default="./out", help="trace 输出目录")
    return p.parse_args()


def make_prompts(engine, prompt_len, num):
    base = engine.tokenizer.encode("The quick brown fox jumps over the lazy dog.")
    token = (base * (prompt_len // len(base) + 1))[:prompt_len]
    return [token.copy() for _ in range(num)]


def phase(engine) -> str:
    """当前 step 的阶段: 有 running 序列则 decode, 否则 prefill。"""
    return "decode" if engine.scheduler.running else "prefill"


def run_and_profile(engine, prompts, sp, iters, trace_dir, tag, row_limit=15):
    """跑一轮生成并剖析, 每步按 prefill/decode 打标签, 返回阶段统计。"""
    schedule = torch.profiler.schedule(wait=0, warmup=0, active=iters, repeat=1)
    stats = {"prefill": 0.0, "decode": 0.0, "steps": 0}

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule,
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function(tag):
            for p in prompts:
                engine.add_request(p, sp)
            while not engine.scheduler.is_finished():
                ph = phase(engine)
                with record_function(ph):
                    engine.step()
                stats[ph] += 1
                stats["steps"] += 1
                prof.step()

    torch.cuda.synchronize()

    # 阶段汇总: 平均每步 CPU / GPU 时间
    print(f"\n=== {tag} ===")
    for ph in ("prefill", "decode"):
        evs = [e for e in prof.events() if e.name == ph]
        if not evs:
            continue
        cpu = sum(e.cpu_time_total for e in evs) / len(evs) / 1000
        gpu = sum(e.device_time_total for e in evs) / len(evs) / 1000
        print(f"  {ph:7s} {len(evs):3d} 步  平均 CPU {cpu:6.1f}ms  平均 GPU {gpu:6.1f}ms")
    print(prof.key_averages().table(sort_by="device_time_total",
                                    row_limit=row_limit))

    os.makedirs(trace_dir, exist_ok=True)
    prof.export_chrome_trace(f"{trace_dir}/{tag}_trace.json")
    print(f"  → {trace_dir}/{tag}_trace.json")
    return stats


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
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    single = make_prompts(engine, args.prompt_len, 1)   # [[...]] 单请求
    batch = make_prompts(engine, args.prompt_len, args.num_reqs)

    # 预热两轮, 不剖析
    preheat = []
    if args.mode in ("single", "both"):
        preheat.append(single)
    if args.mode in ("batch", "both"):
        preheat.append(batch)
    for prompts in preheat:
        for _ in range(args.warm):
            for p in prompts:
                engine.add_request(p, sp)
            while not engine.scheduler.is_finished():
                engine.step()

    if args.mode in ("single", "both"):
        run_and_profile(engine, single, sp, args.max_new_tokens + 1,
                        args.trace_dir, "single", args.row_limit)
    if args.mode in ("batch", "both"):
        run_and_profile(engine, batch, sp, args.num_reqs * args.max_new_tokens,
                        args.trace_dir, "batch", args.row_limit)


if __name__ == "__main__":
    main()
