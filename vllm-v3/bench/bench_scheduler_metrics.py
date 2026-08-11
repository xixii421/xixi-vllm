"""dec@P / short_P 指标实测（v2 vs v3 调度策略对比）。

定义见 vllm-v2/docs/00-v2版本设计.md：

- dec@P  : 一条长 Prefill 请求尚未完成全部 Prefill 的时段（Prefill 窗口）内，
           其他短请求已产出的 decode token 数。越大 → 短请求被阻塞程度越低。
- short_P: 从长请求进入系统排队起，到所有短请求全部完成生成的时间。越小越好。

场景：4 条短请求（prompt=128, max_tokens=8）先入队，随后 1 条长请求
（prompt=3000, max_tokens=4）入队。max_num_batched_tokens=1024，使长请求
prefill 需要 3 个 chunk，prefill 窗口跨多步。

对比矩阵：
- v2 mix_prefill_decode=True  （PD 交错：decode 优先 + prefill 插空）
- v2 mix_prefill_decode=False （prefill-first：旧策略）
- v3 mix_prefill_decode=True
- v3 mix_prefill_decode=False

用法:
    python bench_scheduler_metrics.py --engine v2 --mix true
    python bench_scheduler_metrics.py --engine v3 --mix false
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL = str(Path.home() / "huggingface" / "Qwen3-0.6B")


def make_engine(engine: str, mix: bool):
    sys.path.insert(0, str(ROOT) if engine == "v3" else str(ROOT.parent / "vllm-v2"))
    if engine == "v3":
        from src import Engine, EngineConfig
        from src.sampling_params import SamplingParams

        cfg = EngineConfig(
            model=MODEL,
            enforce_eager=True,
            prefill_cuda_graph=False,
            dtype="auto",
            context_len=4096,
            max_num_seqs=8,
            max_num_batched_tokens=1024,
            mix_prefill_decode=mix,
            kvcache_block_size=256,
            prefix_backend="none",
            gpu_memory_utilization=0.5,
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
            max_num_seqs=8,
            max_num_batched_tokens=1024,
            mix_prefill_decode=mix,
        )
    return Engine(cfg), SamplingParams


def run_scenario(engine, sp_cls) -> dict:
    """执行 4 短 + 1 长的场景，返回 dec@P / short_P 等指标。"""
    short_prompts = [
        [3000 + j * 7 + i for j in range(128)] for i in range(4)  # 4 条短请求
    ]
    long_prompt = [5000 + i % 991 for i in range(3000)]  # 1 条长请求

    # 短请求先入队，长请求后入队（dec@P 度量需要短请求先进入 decode）
    short_seqs = [
        engine.add_request(p, sp_cls(temperature=0.0, max_tokens=8))
        for p in short_prompts
    ]
    long_seq = engine.add_request(
        long_prompt, sp_cls(temperature=0.0, max_tokens=4)
    )

    t0 = time.perf_counter()
    window_started = False
    recorded = False
    base_short_comp = 0
    dec_at_p = 0
    long_ttft = None
    short_done_at = None
    all_done_at = None

    while not engine.scheduler.is_finished():
        engine.step()
        now = time.perf_counter() - t0

        # 长请求 prefill 窗口：从首次被调度 prefill 到 prefill 完成
        if not recorded:
            if not window_started:
                if long_seq.num_cached_tokens > 0:
                    window_started = True
                    base_short_comp = sum(
                        s.num_completion_tokens for s in short_seqs
                    )
            elif long_seq.num_cached_tokens >= long_seq.num_prompt_tokens:
                # 窗口结束：dec@P = 窗口内短请求新增 completion tokens
                dec_at_p = (
                    sum(s.num_completion_tokens for s in short_seqs)
                    - base_short_comp
                )
                recorded = True

        if long_ttft is None and long_seq.num_completion_tokens >= 1:
            long_ttft = now * 1000

        if short_done_at is None and all(
            s.status.name == "FINISHED" for s in short_seqs
        ):
            short_done_at = now * 1000

    all_done_at = (time.perf_counter() - t0) * 1000
    total_completion = sum(
        s.num_completion_tokens for s in short_seqs + [long_seq]
    )
    return {
        "dec_at_p": dec_at_p,
        "short_p_ms": short_done_at or all_done_at,
        "all_done_ms": all_done_at,
        "long_ttft_ms": long_ttft,
        "total_completion": total_completion,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=("v2", "v3"), required=True)
    p.add_argument("--mix", choices=("true", "false"), required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.cuda.set_device(args.device)
    mix = args.mix == "true"

    engine, sp_cls = make_engine(args.engine, mix)
    print(f"[info] engine={args.engine} mix_prefill_decode={mix} ready")

    # warmup 编译
    engine.add_request([7, 13, 29, 31, 47], sp_cls(temperature=0.0, max_tokens=1))
    while not engine.scheduler.is_finished():
        engine.step()
    engine.reset_kv()

    r = run_scenario(engine, sp_cls)
    print(f"[result] engine={args.engine} mix={mix}: dec@P={r['dec_at_p']} "
          f"short_P={r['short_p_ms']:.1f}ms all_done={r['all_done_ms']:.1f}ms "
          f"long_TTFT={r['long_ttft_ms']:.1f}ms total_completion={r['total_completion']}")

    if hasattr(engine, "destroy"):
        engine.destroy()
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    out = Path(__file__).resolve().parent / "out"
    out.mkdir(parents=True, exist_ok=True)
    fname = f"scheduler_metrics_{args.engine}_{args.mix}.md"
    lines = [
        f"# dec@P / short_P — engine={args.engine} mix_prefill_decode={mix}",
        "",
        f"- dec@P: {r['dec_at_p']}（长请求 prefill 窗口内短请求 decode token 数）",
        f"- short_P: {r['short_p_ms']:.1f} ms",
        f"- 全部完成: {r['all_done_ms']:.1f} ms",
        f"- 长请求 TTFT: {r['long_ttft_ms']:.1f} ms",
        f"- 总产出 tokens: {r['total_completion']}",
    ]
    (out / fname).write_text("\n".join(lines) + "\n")
    print(f"\n→ saved to {out / fname}")


if __name__ == "__main__":
    main()
