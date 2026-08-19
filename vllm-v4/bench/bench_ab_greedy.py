"""引擎级 A/B：argmax 快路径 vs 全量 logits 贪心，贪婪批量 decode 吞吐对比。

默认（fast）：ModelRunner 走 ParallelLMHead.argmax。
模拟旧行为（old）：把 argmax monkeypatch 成全量 logits 后取 argmax。

用法（在 vllm-v4/ 下）:
    ../.venv2/bin/python bench/bench_ab_greedy.py --mode fast
    ../.venv2/bin/python bench/bench_ab_greedy.py --mode old
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCH))

MODEL = str(Path.home() / "huggingface" / "Qwen3-0.6B")
PROMPT_LEN = 128
MAX_TOKENS = 96
N_REQS = 8


def dummy_ids(n: int, salt: int = 0) -> list[int]:
    n = max(1, n)
    return [((i * 7919 + 10007 + salt * 97) % 150000) + 100 for i in range(n)]


def main() -> None:
    from src import Engine, EngineConfig, SamplingParams  # noqa: E402

    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("fast", "old"), default="fast")
    args = p.parse_args()

    if args.mode == "old":
        from src.compute.models.qwen3 import Qwen3ForCausalLM

        # 旧行为：全量 logits 拼接后 argmax（保留原 all_gather + 截断 + 采样语义）
        Qwen3ForCausalLM.argmax = lambda self, h: self.compute_logits(h).argmax(-1)
        tag = "old(全量 logits)"
    else:
        tag = "fast(argmax 快路径)"

    cfg = EngineConfig(
        model=MODEL,
        enforce_eager=False,  # decode CUDA Graph
        torch_compile=False,
        dtype="auto",
        context_len=4096,
        max_num_seqs=N_REQS,
        max_num_batched_tokens=4096,
        mix_prefill_decode=True,
        kvcache_block_size=256,
        prefix_backend="none",
        gpu_memory_utilization=0.85,
    )
    engine = Engine(cfg)

    sp_w = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
    for it in range(2):
        for _ in range(N_REQS):
            seq = engine.add_request(dummy_ids(PROMPT_LEN, salt=it), sp_w)
        while not engine.scheduler.is_finished():
            engine.step()
    torch.cuda.synchronize()

    sp = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=MAX_TOKENS)
    seqs = [engine.add_request(dummy_ids(PROMPT_LEN, salt=100 + i), sp)
            for i in range(N_REQS)]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while not engine.scheduler.is_finished():
        engine.step()
    torch.cuda.synchronize()
    e2e = time.perf_counter() - t0

    n = sum(len(s.token_ids) - s.num_prompt_tokens for s in seqs)
    print(f"[{tag}] {N_REQS} reqs x {MAX_TOKENS} tok: "
          f"{n} completion tokens in {e2e:.3f}s ({n / e2e:.1f} tok/s)")
    engine.destroy()


if __name__ == "__main__":
    main()
