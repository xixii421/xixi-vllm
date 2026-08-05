"""决定性实验：不预热时 radix 的 dedup 是否真的优于 hash。

场景：20 用户同时提交相同 512-tok 前缀（各自不同 tail），KV 池只有 40 块。
  · hash：首轮全部 miss → 每人独占 2 块前缀 → 40 块前缀 + tail 严重超池 → 抢占/失败
  · radix：publish 时 dedup remap 到 canonical 块 → 前缀只占 2 块 → 能全部放下

观察：完成率、e2e、cached 块数、是否触发抢占（KV 池压力）。
"""
from __future__ import annotations

import gc
import os
import random
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import EngineConfig
from src.engine.engine import Engine
from src.sampling_params import SamplingParams

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B")


def make_prompt(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(0, 5000) for _ in range(n)]


def run(prefix: str, *, n_users: int, prefix_tok: int, tail_tok: int, kv_blocks: int) -> dict:
    cfg = EngineConfig(
        model=MODEL,
        context_len=4096,
        max_num_batched_tokens=8192,
        prefix_backend=prefix,
        num_kvcache_blocks=kv_blocks,
        enforce_eager=True,
        mix_prefill_decode=True,
        prefill_cuda_graph=False,
    )
    t0 = time.perf_counter()
    eng = Engine(cfg)
    init_s = time.perf_counter() - t0
    sp = SamplingParams(temperature=0.0, max_tokens=8)

    sys_prefix = make_prompt(prefix_tok, seed=7)
    seqs = [eng.add_request(sys_prefix + make_prompt(tail_tok, seed=100 + u), sp)
            for u in range(n_users)]

    t0 = time.monotonic()
    stuck = False
    try:
        while not all(s.is_finished for s in seqs):
            eng.step()
            if time.monotonic() - t0 > 60:
                stuck = True
                break
    except RuntimeError as e:
        return {"error": str(e)[:100], "init_s": init_s}

    e2e = (time.monotonic() - t0) * 1000
    done = sum(1 for s in seqs if s.is_finished)
    kv = {"free": eng.kv.num_free, "cached": eng.kv.blocks.num_cached,
          "used": len(eng.kv.blocks.used_ids)}
    eng.destroy(); del eng; gc.collect(); torch.cuda.empty_cache()
    return {"init_s": init_s, "e2e_ms": e2e, "done": done, "total": n_users,
            "stuck": stuck, "kv": kv}


def main() -> None:
    torch.cuda.empty_cache()
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n=== 决定性实验：20 用户相同 512-tok 前缀，KV 池 40 块（紧张）===")
    for name, backend in [("hash", "hash"), ("radix", "radix"), ("none", "none")]:
        r = run(backend, n_users=20, prefix_tok=512, tail_tok=64, kv_blocks=40)
        if "error" in r:
            print(f"[{name:<5}] ERROR: {r['error']}")
            continue
        print(f"[{name:<5}] init={r['init_s']:.1f}s 完成={r['done']}/{r['total']} "
              f"e2e={r['e2e_ms']:>6.0f}ms stuck={r['stuck']} "
              f"kv_free={r['kv']['free']:>3} cached={r['kv']['cached']:>3} used={r['kv']['used']:>3}")

    print("\n=== 对照：更紧张（30 用户 × 512 前缀，KV 池 40 块）===")
    for name, backend in [("hash", "hash"), ("radix", "radix")]:
        r = run(backend, n_users=30, prefix_tok=512, tail_tok=64, kv_blocks=40)
        if "error" in r:
            print(f"[{name:<5}] ERROR: {r['error']}")
            continue
        print(f"[{name:<5}] init={r['init_s']:.1f}s 完成={r['done']}/{r['total']} "
              f"e2e={r['e2e_ms']:>6.0f}ms stuck={r['stuck']} "
              f"kv_free={r['kv']['free']:>3} cached={r['kv']['cached']:>3} used={r['kv']['used']:>3}")


if __name__ == "__main__":
    main()
