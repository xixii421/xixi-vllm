"""TP 下 CUDA Graph (CG) vs eager 的完整对比。

每个配置（--tp × --cg/--eager）测四组场景：
  - 单请求 TTFT（prompt=256）
  - 单请求 decode 吞吐（128 tokens）
  - batch=4 / 8 / 16 并发 decode 聚合吞吐
  - 每组场景记录 decode 单步墙钟延迟分布（中位数 / p95）

用法（在 vllm-v4/ 下）:
    ../.venv2/bin/python bench/bench_cg_vs_eager.py --tp 2 --tag tp2-eager
    ../.venv2/bin/python bench/bench_cg_vs_eager.py --tp 2 --cg --tag tp2-cg
    ../.venv2/bin/python bench/bench_cg_vs_eager.py --tp 1 --tag tp1-eager
    ../.venv2/bin/python bench/bench_cg_vs_eager.py --tp 1 --cg --tag tp1-cg
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import Engine, EngineConfig, SamplingParams  # noqa: E402

MODEL = str(Path.home() / "huggingface" / "Qwen3-0.6B")
OUT = Path(__file__).resolve().parent / "out"
PROMPT_LEN = 256
MAX_TOKENS = 256
MAX_BS = 16


def dummy_ids(n: int, salt: int = 0) -> list[int]:
    n = max(1, n)
    return [((i * 7919 + 10007 + salt * 97) % 150000) + 100 for i in range(n)]


def make_engine(tp: int, model: str = MODEL, *, cg: bool) -> Engine:
    return Engine(
        EngineConfig(
            model=model,
            tp_size=tp,
            pp_size=1,
            enforce_eager=not cg,
            torch_compile=False,
            dtype="bfloat16",
            context_len=4096,
            max_num_seqs=MAX_BS,
            max_num_batched_tokens=4096,
            prefix_backend="none",
            gpu_memory_utilization=0.6,
            num_kvcache_blocks=64,
        )
    )


def measure(
    engine: Engine, n: int, *, prompt_len: int, max_tokens: int
) -> dict:
    """跑一批请求，返回 TTFT、decode tok/s、decode 单步延迟列表。"""
    sps = [
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=max_tokens)
        for _ in range(n)
    ]
    seqs = [
        engine.add_request(dummy_ids(prompt_len, salt=1000 + i), sps[i])
        for i in range(n)
    ]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ttft = None
    step_lat: list[float] = []
    while not engine.scheduler.is_finished():
        ts = time.perf_counter()
        engine.step()
        step_lat.append(time.perf_counter() - ts)
        if ttft is None and any(s.num_completion_tokens > 0 for s in seqs):
            torch.cuda.synchronize()
            ttft = time.perf_counter() - t0
    torch.cuda.synchronize()
    e2e = time.perf_counter() - t0
    n_out = sum(s.num_completion_tokens for s in seqs)
    # decode 吞吐：去掉每个请求的第一个 token（TTFT 已计）
    decode_tps = (n_out - n) / (e2e - ttft) if (e2e - ttft) > 0 else 0.0
    return {
        "ttft_ms": ttft * 1000 if ttft is not None else float("nan"),
        "decode_tps": decode_tps,
        "n_out": n_out,
        "steps": step_lat,
    }


def report_scenario(engine: Engine, tag: str, n: int) -> None:
    r = measure(engine, n, prompt_len=PROMPT_LEN, max_tokens=MAX_TOKENS)
    steps = r["steps"]
    # decode 步 = 总步数去掉 prefill 起步阶段；这里用延迟的 p95/中位数描述整体
    med = statistics.median(steps) if steps else float("nan")
    p95 = (
        sorted(steps)[int(len(steps) * 0.95) - 1] if len(steps) >= 20 else float("nan")
    )
    print(
        f"{tag} | n={n:>2} | TTFT {r['ttft_ms']:7.2f} ms | "
        f"decode {r['decode_tps']:7.1f} tok/s | step_med {med * 1e3:6.2f} ms | "
        f"step_p95 {p95 * 1e3:6.2f} ms | steps {len(steps)}"
    )
    return r


def run(tp: int, tag: str, *, cg: bool, model: str = MODEL) -> None:
    print(f"=== {tag}: tp={tp} cg={cg} ===", flush=True)
    engine = make_engine(tp, model, cg=cg)

    # 预热（prompt_len 等长，让 CG 捕获 / eager 稳定）
    sp_w = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
    for it in range(3):
        seq = engine.add_request(dummy_ids(PROMPT_LEN, salt=it), sp_w)
        while not seq.is_finished:
            engine.step()
    torch.cuda.synchronize()

    results = []
    for n in [1, 4, 8, 16]:
        results.append(report_scenario(engine, tag, n))

    engine.destroy()
    # 汇总写文件
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "bench_cg_vs_eager.txt"
    first = not path.exists() or path.stat().st_size == 0
    with path.open("a") as f:
        if first:
            f.write("config | n | TTFT_ms | decode_tps | step_med_ms | step_p95_ms\n")
        for n, r in zip([1, 4, 8, 16], results):
            med = statistics.median(r["steps"]) if r["steps"] else float("nan")
            p95 = (
                sorted(r["steps"])[int(len(r["steps"]) * 0.95) - 1]
                if len(r["steps"]) >= 20
                else float("nan")
            )
            f.write(
                f"{tag} | {n} | {r['ttft_ms']:.2f} | {r['decode_tps']:.1f} | "
                f"{med * 1e3:.2f} | {p95 * 1e3:.2f}\n"
            )
    print(f"→ {path}")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--tag", default="cfg")
    p.add_argument("--cg", action="store_true", help="启用 decode CUDA Graph")
    p.add_argument("--model", default=MODEL)
    args = p.parse_args()
    run(args.tp, args.tag, cg=args.cg, model=str(Path(args.model).expanduser()))


if __name__ == "__main__":
    main()
