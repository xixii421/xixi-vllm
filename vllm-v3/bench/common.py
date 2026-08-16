"""bench 公共件：模型路径、引擎工厂、等长 dummy、drain / warmup / 销毁。"""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import Engine, EngineConfig, SamplingParams  # noqa: E402

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B")
WARMUP_ITERS = 3  # reduce-overhead：编译 / 录图 / 重放，计时从下一轮开始
OUT_DIR = Path(__file__).resolve().parent / "out"


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


def warmup(engine: Engine, n: int, *, iters: int = WARMUP_ITERS, max_tokens: int = 2) -> None:
    """对本 prompt 长度跑 compile + 录图；greedy，避免和采样冷启动缠在一起。"""
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    torch.cuda.synchronize()
    for it in range(iters):
        engine.add_request(dummy_ids(n, salt=it), sp)
        drain(engine)
    torch.cuda.synchronize()


def ttft_until_first(engine: Engine, token_ids: list[int]) -> float:
    """add 之后到第一个 completion token（greedy, max_tokens=1）。调用方先 warmup。"""
    sp = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    seq = engine.add_request(token_ids, sp)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while not seq.is_finished:
        engine.step()
        if seq.num_completion_tokens > 0:
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def destroy(engine: Engine | None) -> None:
    if engine is None:
        return
    engine.destroy()
    gc.collect()
    torch.cuda.empty_cache()


def median(xs: list[float]) -> float:
    ys = sorted(xs)
    return ys[len(ys) // 2]


def write_result(name: str, body: str) -> Path:
    """写入 bench/out/<name>，覆盖旧文件。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    text = body if body.endswith("\n") else body + "\n"
    path.write_text(text)
    print(f"→ {path}")
    return path


def append_lines(name: str, lines: list[str], *, header: str | None = None) -> Path:
    """追加多行到 bench/out/<name>（txt）。

    header 仅在文件为空时写入一次；跨引擎脚本各自 append 自己的行，
    避免 JSON 中间文件与合并逻辑。
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    first = not path.exists() or path.stat().st_size == 0
    with path.open("a") as f:
        if first and header:
            f.write(header.rstrip("\n") + "\n")
        for ln in lines:
            f.write(ln.rstrip("\n") + "\n")
    print(f"→ {path}")
    return path
