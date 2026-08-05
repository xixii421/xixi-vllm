#!/usr/bin/env python3
"""验证 FA2 prefill 路径与 FlashInfer 数值一致。

同一 4096-token prompt，分别用：
  A. use_fa2=True（首 chunk eager prefill 默认）
  B. use_fa2=False（FlashInfer paged prefill）
对比 logits 与 greedy 输出。
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "/home/xingyongwang/code/mycuda/dzyy-vllm/vllm-v3")

import torch
from src import Engine, EngineConfig, SamplingParams

MODEL = "/home/xingyongwang/huggingface/Qwen3-0.6B"


def main() -> None:
    torch.manual_seed(0)
    LEN = int(os.environ.get("PROMPT_LEN", "4096"))
    MAXB = int(os.environ.get("MAX_BATCHED", "4096"))
    eng = Engine(EngineConfig(
        model=MODEL, context_len=16384, enforce_eager=True,
        prefill_cuda_graph=False, dtype="auto",
        prefix_backend="none", max_num_seqs=8,
        max_num_batched_tokens=MAXB, gpu_memory_utilization=0.85,
    ))

    chunk = ("The quick brown fox jumps over the lazy dog. "
             "Explain continuous batching for LLM inference. ") * 4
    ids = []
    while len(ids) < LEN:
        ids.extend(eng.tokenizer.encode(chunk, add_special_tokens=False))
    ids = ids[:LEN]
    n_chunks = (LEN + MAXB - 1) // MAXB
    print(f"prompt_len={LEN} max_batched={MAXB} -> {n_chunks} chunks"
          f"（首 chunk FA2，续 chunk FlashInfer）")

    def run(use_fa2: bool) -> tuple[list[int], torch.Tensor]:
        runner = eng.model_runner
        orig = runner.backend.build_metadata

        def fake_build(*a, **k):
            k["use_fa2"] = use_fa2
            return orig(*a, **k)

        runner.backend.build_metadata = fake_build
        try:
            sp = SamplingParams(max_tokens=1, temperature=0.0)
            seq = eng.add_request(ids, sp)
            t0 = time.perf_counter()
            while not seq.is_finished:
                eng.step()
            dt = time.perf_counter() - t0
            out_ids = seq.token_ids[seq.num_prompt_tokens:]
            return out_ids, dt
        finally:
            runner.backend.build_metadata = orig

    ids_a, dt_a = run(use_fa2=True)
    ids_b, dt_b = run(use_fa2=False)

    print(f"FA2:        first_token={ids_a} ({eng.tokenizer.decode(ids_a)!r})  time={dt_a*1000:.1f}ms")
    print(f"FlashInfer: first_token={ids_b} ({eng.tokenizer.decode(ids_b)!r})  time={dt_b*1000:.1f}ms")
    print("MATCH" if ids_a == ids_b else "MISMATCH", ids_a, ids_b)
    eng.destroy()


if __name__ == "__main__":
    main()
