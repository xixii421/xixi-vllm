"""Piecewise Prefill CUDA Graph 实测：eager vs prefill-CG 的 TTFT 对比。

对 v3 的 `prefill_cuda_graph` 特性做真实测量。piece CG 只对「首 chunk 且
piece 的 replay GPU 时间 < eager 发射地板」有效（Qwen3-0.6B 上 ≤~1536 token），
因此用短-中长度 prompt 的 TTFT 对比最能体现收益。

TTFT = add_request 到第一个 completion token 产出（含调度 + prefill）。

注意：单引擎顺序测多个长度（KV 池足够大，旧请求完成后块被释放复用），
避免反复 reset_kv（会重建 ModelRunner 并重新 capture 全部 piece graphs）。

用法:
    python bench_prefill_cg.py                     # 全部长度
    python bench_prefill_cg.py --only 256,1024
    python bench_prefill_cg.py --device cuda:0
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import Engine, EngineConfig, SamplingParams  # noqa: E402
from src.models.hub import DEFAULT_CACHE_DIR  # noqa: E402

MODEL = str(DEFAULT_CACHE_DIR / "Qwen3-0.6B")


def make_engine(*, prefill_cg: bool) -> Engine:
    cfg = EngineConfig(
        model=MODEL,
        enforce_eager=True,  # 关闭 decode CG，只隔离 prefill 变量
        prefill_cuda_graph=prefill_cg,
        dtype="auto",
        context_len=4096,
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        mix_prefill_decode=True,
        kvcache_block_size=256,
        prefix_backend="none",
        gpu_memory_utilization=0.5,
        prefill_cg_max_tokens=1536,
    )
    return Engine(cfg)


def ttft(engine: Engine, token_ids: list[int]) -> float:
    """单请求 TTFT：add_request → 第一个完成请求（max_tokens=1 时即首次 decode）。"""
    engine.add_request(token_ids, SamplingParams(temperature=0.0, max_tokens=1))
    t0 = time.perf_counter()
    while not engine.scheduler.is_finished():
        outputs, _ = engine.step()
        if outputs:
            return (time.perf_counter() - t0) * 1000
    return (time.perf_counter() - t0) * 1000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="", help="逗号分隔的 prompt 长度")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.cuda.set_device(args.device)

    lengths = [128, 256, 512, 1024, 1536] if not args.only else [
        int(x) for x in args.only.split(",")
    ]

    print(f"# Piecewise Prefill CUDA Graph — TTFT 对比（{args.device}）\n")
    print("| prompt len | eager (ms) | prefill-CG (ms) | CG 加速 |")
    print("| --- | ---: | ---: | ---: |")

    rows = []
    for prefill_cg in (False, True):
        engine = make_engine(prefill_cg=prefill_cg)
        n_pieces = (
            len(engine.model_runner.prefill_cg_runner.pieces)
            if engine.model_runner.prefill_cg_runner
            else 0
        )
        print(f"[info] engine prefill_cg={prefill_cg} ready "
              f"({n_pieces} pieces captured if CG)")
        # 触发一次最短 prefill 的编译/autotune，避免首轮污染计时
        ttft(engine, [7, 13, 29, 31, 47, 53, 59, 61])
        for i, L in enumerate(lengths):
            tok = [100 + i % 1000 for i in range(L)]
            t = ttft(engine, tok)
            if prefill_cg:
                rows[i] = (rows[i][0], rows[i][1], t)
            else:
                rows.append((L, t, 0.0))
            print(f"[{prefill_cg and 'CG' or 'eager'}] L={L}: {t:.2f} ms", flush=True)
        engine.destroy()
        del engine
        gc.collect()
        torch.cuda.empty_cache()

    print("")
    print("| prompt len | eager (ms) | prefill-CG (ms) | CG 加速 |")
    print("| --- | ---: | ---: | ---: |")
    for L, t_e, t_c in rows:
        ratio = t_e / t_c if t_c > 0 else float("inf")
        print(f"| {L} | {t_e:.2f} | {t_c:.2f} | {ratio:.2f}x |")

    out = Path(__file__).resolve().parent / "out"
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Piecewise Prefill CUDA Graph — TTFT 对比", ""]
    lines += ["| prompt len | eager (ms) | prefill-CG (ms) | CG 加速 |",
              "| --- | ---: | ---: | ---: |"]
    for L, t_e, t_c in rows:
        lines.append(
            f"| {L} | {t_e:.2f} | {t_c:.2f} | {t_e / t_c if t_c > 0 else float('inf'):.2f}x |"
        )
    (out / "prefill_cg_ttft.md").write_text("\n".join(lines) + "\n")
    print(f"\n→ saved to {out / 'prefill_cg_ttft.md'}")


if __name__ == "__main__":
    main()
