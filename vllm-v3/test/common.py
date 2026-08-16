"""test 公共件：引擎工厂、等长 dummy、greedy 运行与销毁。

只做功能正确性，不含计时逻辑。
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import Engine, EngineConfig, SamplingParams  # noqa: E402

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B")


def dummy_ids(n: int, salt: int = 0) -> list[int]:
    n = max(1, n)
    return [((i * 7919 + 10007 + salt * 97) % 150000) + 100 for i in range(n)]


def make_engine(
    *,
    cg: bool = False,
    compile: bool = False,
    prefix: str = "none",
    mix: bool = True,
    max_num_seqs: int = 64,
    max_batched: int = 8192,
    gpu_mem: float = 0.85,
    context_len: int = 4096,
) -> Engine:
    return Engine(
        EngineConfig(
            model=MODEL,
            enforce_eager=not cg,
            torch_compile=compile,
            compile_dynamic=True,
            dtype="auto",
            context_len=context_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_batched,
            mix_prefill_decode=mix,
            kvcache_block_size=256,
            prefix_backend=prefix,
            gpu_memory_utilization=gpu_mem,
        )
    )


def drain(engine: Engine) -> None:
    while not engine.scheduler.is_finished():
        engine.step()


def warmup(engine: Engine, n: int, *, iters: int = 3, max_tokens: int = 2) -> None:
    """跑若干轮短请求，让 torch.compile 与 CUDA Graph 就绪。"""
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    torch.cuda.synchronize()
    for it in range(iters):
        engine.add_request(dummy_ids(n, salt=it), sp)
        drain(engine)
    torch.cuda.synchronize()


def greedy_ids(engine: Engine, prompt: list[int], max_tokens: int) -> list[int]:
    """跑完整条请求，返回 completion token ids。"""
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    seq = engine.add_request(prompt, sp)
    while not seq.is_finished:
        engine.step()
    return list(seq.token_ids[seq.num_prompt_tokens:])


def destroy(engine: Engine | None) -> None:
    if engine is None:
        return
    engine.destroy()
    gc.collect()
    torch.cuda.empty_cache()
