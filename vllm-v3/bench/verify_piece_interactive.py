#!/usr/bin/env python3
"""piece CUDA Graph 交互式验证。

并排启动两个引擎（prefill CG 开 / 关），对同一批输入实测：

  · 是否命中 piece CG（哪个 (bs,L) 分片）
  · 命中时 TTFT 加速多少（vs eager）
  · 未命中（长文本/网格外）时行为
  · 输出 token 与 eager 是否一致（数值正确性）

用法：
  # 交互模式：逐行输入文本，回车即测
  CUDA_VISIBLE_DEVICES=0 python bench/verify_piece_interactive.py --interactive

  # 自动模式：扫一组典型长度
  CUDA_VISIBLE_DEVICES=0 python bench/verify_piece_interactive.py --lens 129,256,513,1024,2000

  # 长文本对照（应回退 eager）
  CUDA_VISIBLE_DEVICES=0 python bench/verify_piece_interactive.py --lens 1600,2000,3000
"""
from __future__ import annotations

import argparse
import gc
import os
import random
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import EngineConfig
from src.engine.engine import Engine
from src.sampling_params import SamplingParams

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B")
KV_BLOCKS = 96
GEN_TOKENS = 1  # 只验证 prefill TTFT 时生成 1 token


def make_engine(prefill_cg: bool) -> Engine:
    cfg = EngineConfig(
        model=MODEL,
        context_len=4096,
        max_num_batched_tokens=8192,
        prefix_backend="none",            # 关前缀缓存，只测 piece CG
        num_kvcache_blocks=KV_BLOCKS,
        enforce_eager=True,               # 不启用 decode CG，隔离 piece CG 效果
        mix_prefill_decode=True,
        prefill_cuda_graph=prefill_cg,
    )
    return Engine(cfg)


def run_one(eng: Engine, prompt_ids: list[int], gen: int = GEN_TOKENS) -> tuple[float, list[int]]:
    """跑一个请求，返回 (TTFT_ms, 生成 token 列表)。"""
    sp = SamplingParams(temperature=0.0, max_tokens=gen, top_p=None, top_k=None)
    seq = eng.add_request(prompt_ids, sp)
    first = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while not seq.is_finished:
        eng.step()
        if first is None and seq.num_completion_tokens > 0:
            torch.cuda.synchronize()
            first = time.perf_counter() - t0
    torch.cuda.synchronize()
    ttft = (first if first is not None else time.perf_counter() - t0) * 1000
    toks = list(seq.token_ids[seq.num_prompt_tokens :])
    return ttft, toks


def predict_piece(eng: Engine, n: int) -> tuple[str, int]:
    """预测该长度会命中哪个 piece，或回退 eager。

    与 runner.can_use 同逻辑：找最小覆盖 L（贪心）。这里只显示（bs=1 且 n ≤ 1536
    且 piece 实测 GPU 时间 < eager 地板才真命中；bs=1 网格内通常满足）。
    """
    r = eng.model_runner.prefill_cg_runner
    if r is None or not r._captured:
        return "no-cg", 0
    best = None
    for pbs, L in r.pieces:
        if pbs == 1 and L >= n:
            if best is None or L < best:
                best = L
    if best is None:
        return "miss", 0
    if r._piece_gpu_ms.get((1, best), float("inf")) < r._eager_floor_ms:
        return "hit", best
    return "miss", best


def make_prompt(eng: Engine, n: int, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(0, 5000) for _ in range(n)]


def compare(eng_cg: Engine, eng_eag: Engine, prompt_ids: list[int], label: str) -> None:
    n = len(prompt_ids)
    r = eng_cg.model_runner.prefill_cg_runner
    real = r.can_use(1, [n]) if (r is not None and r._captured) else None
    kind, L = predict_piece(eng_cg, n)
    if real is not None:
        kind, L = "hit", real[1]          # 以引擎真实判定为准

    # warmup 一次（编译/autotune 不计时），再测 3 次取中位
    run_one(eng_cg, prompt_ids)
    run_one(eng_eag, prompt_ids)
    cg_t, eag_t = [], []
    cg_out = eag_out = None
    for _ in range(3):
        t, o = run_one(eng_cg, prompt_ids)
        cg_t.append(t)
        cg_out = o
        t, o = run_one(eng_eag, prompt_ids)
        eag_t.append(t)
        eag_out = o
    cg_ms = sorted(cg_t)[1]
    eag_ms = sorted(eag_t)[1]
    same = cg_out == eag_out

    if kind == "hit":
        speedup = eag_ms / cg_ms if cg_ms > 0 else float("inf")
        tag = f"HIT  piece=(1,{L})  pad={L/n:.2f}x"
    elif kind == "miss":
        tag = "MISS (>1536 网格外 → eager 回退)"
    else:
        tag = "no-cg engine"

    print(f"\n{'─' * 78}")
    print(f"prompt_len={n:<5d} | {tag}")
    print(f"  {'CG   ':8s} TTFT={cg_ms:7.2f} ms   out={cg_out}")
    print(f"  {'EAGER':8s} TTFT={eag_ms:7.2f} ms   out={eag_out}")
    if kind == "hit":
        print(f"  >> 加速 {speedup:.2f}x" + ("  ✔ 输出一致" if same else "  ✘ 输出不一致!"))
    else:
        print(f"  >> 回退 eager，与 eager 等速" + ("  ✔ 输出一致" if same else "  ✘ 输出不一致!"))
    print(f"{'─' * 78}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactive", action="store_true", help="交互模式：逐行输入文本")
    ap.add_argument("--lens", type=str, default="", help="自动模式：逗号分隔长度")
    args = ap.parse_args()

    t0 = time.perf_counter()
    eng_cg = make_engine(prefill_cg=True)
    t_cg = time.perf_counter() - t0
    t0 = time.perf_counter()
    eng_eag = make_engine(prefill_cg=False)
    t_eag = time.perf_counter() - t0
    print(f"引擎初始化：CG={t_cg:.1f}s  eager={t_eag:.1f}s")

    r = eng_cg.model_runner.prefill_cg_runner
    bs1_ls = sorted({L for b, L in r.pieces if b == 1})
    print(f"bs=1 piece 网格: {bs1_ls}")
    print(f">1536 token 的请求将回退 eager（GPU-bound，CG 无益）")

    try:
        if args.interactive:
            print("\n【交互模式】输入一行文本，回车即测 TTFT 与 CG 命中。输入 quit 退出。")
            while True:
                line = input("\n> ").strip()
                if not line or line in ("quit", "exit", "q"):
                    break
                ids = eng_cg.tokenizer.encode(line)
                print(f"  tokenized → {len(ids)} tokens")
                compare(eng_cg, eng_eag, ids, "interactive")
        else:
            lens = [int(x) for x in args.lens.split(",") if x.strip()]
            if not lens:
                lens = [129, 256, 513, 1024, 2000]
            print("\n【自动模式】")
            for n in lens:
                compare(eng_cg, eng_eag, make_prompt(eng_cg, n), f"len={n}")
    finally:
        eng_cg.destroy()
        eng_eag.destroy()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
