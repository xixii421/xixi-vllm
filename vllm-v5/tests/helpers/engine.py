"""Shared helpers for pytest and scripts.

Keep the one-shot wrappers for compatibility, but also expose session helpers so
bench scripts can reuse a loaded engine across multiple requests.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import gc
import os
import time
from typing import Any, Iterator

import torch
from PIL import Image

from src import Engine, EngineConfig, SamplingParams

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def free_cuda() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def cfg(model: str | os.PathLike[str], **kwargs: Any) -> EngineConfig:
    defaults: dict[str, Any] = dict(
        model=str(model),
        enforce_eager=True,
        max_num_seqs=4,
        context_len=1024,
        num_kvcache_blocks=32,
        gpu_memory_utilization=0.75,
        nccl_port=0,
        enable_vit_cuda_graph=False,
        rpc_backend="zmq",
    )
    defaults.update(kwargs)
    return EngineConfig(**defaults)


def solid_image(size: tuple[int, int] = (64, 64), color=(32, 128, 200)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def homo_device_ids(n: int, *, prefer: str = "3090") -> list[int]:
    """Pick ``n`` same-SKU GPUs (prefer name substring). Raises if not enough."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    groups: dict[str, list[int]] = {}
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_properties(i).name
        groups.setdefault(name, []).append(i)
    preferred = [g for name, g in groups.items() if prefer.lower() in name.lower()]
    candidates = preferred + [g for name, g in groups.items() if prefer.lower() not in name.lower()]
    for g in candidates:
        if len(g) >= n:
            return g[:n]
    raise RuntimeError(
        f"need {n} homogeneous GPUs; groups={[ (k,v) for k,v in groups.items() ]}"
    )


def tp_divisible(
    model: str | os.PathLike[str],
    tp: int,
) -> tuple[bool, str]:
    """Whether model dimensions allow this TP size."""
    from tests.helpers.parallel import tp_divisible as _tp_divisible

    return _tp_divisible(model, tp)


@contextmanager
def engine_session(
    config: EngineConfig,
    *,
    clear_cuda: bool = True,
) -> Iterator[Engine]:
    if clear_cuda:
        free_cuda()
    eng = Engine(config)
    try:
        yield eng
    finally:
        eng.destroy()
        if clear_cuda:
            free_cuda()


def greedy_with_engine(
    eng: Engine,
    prompt: str | list[int],
    *,
    n: int = 8,
    images=None,
) -> dict[str, Any]:
    seq = eng.add_request(
        prompt,
        SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True),
        images=images,
    )
    while not seq.is_finished:
        eng.step()
    return {
        "tokens": seq.token_ids[seq.num_prompt_tokens :],
        "rope_delta": int(seq.rope_delta),
        "mrope_next": seq.mrope_next,
        "origins": [mf.mrope_origin for mf in seq.mm_features],
    }


def greedy(
    config: EngineConfig,
    prompt: str | list[int],
    *,
    n: int = 8,
    images=None,
) -> dict[str, Any]:
    with engine_session(config) as eng:
        return greedy_with_engine(eng, prompt, n=n, images=images)


def greedy_tokens_with_engine(
    eng: Engine,
    prompt: str | list[int],
    *,
    n: int = 8,
    images=None,
) -> list[int]:
    return greedy_with_engine(eng, prompt, n=n, images=images)["tokens"]


def greedy_tokens(
    config: EngineConfig,
    prompt: str | list[int],
    *,
    n: int = 8,
    images=None,
) -> list[int]:
    return greedy(config, prompt, n=n, images=images)["tokens"]


def measure_with_engine(
    eng: Engine,
    config: EngineConfig,
    prompt: str | list[int],
    *,
    n: int = 64,
    warmup: int = 2,
    repeats: int = 3,
) -> dict[str, Any]:
    """Warmup + timed greedy decode. Returns tokens/throughput."""
    for _ in range(warmup):
        seq = eng.add_request(
            prompt, SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
        )
        while not seq.is_finished:
            eng.step()
    times: list[float] = []
    ttfts: list[float] = []
    last: list[int] = []
    for _ in range(repeats):
        seq = eng.add_request(
            prompt, SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
        )
        t0 = time.perf_counter()
        first = None
        while not seq.is_finished:
            eng.step()
            if first is None and seq.num_completion_tokens >= 1:
                first = time.perf_counter()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        ttfts.append((first - t0) * 1000.0 if first else float("nan"))
        last = seq.token_ids[seq.num_prompt_tokens :]
    e2e = sum(times) / len(times)
    ttft = sum(ttfts) / len(ttfts)
    decode_s = max(e2e - ttft / 1000.0, 1e-6)
    return {
        "tokens": last,
        "e2e_s": e2e,
        "ttft_ms": ttft,
        "e2e_tok_s": n / e2e,
        "decode_tok_s": (n - 1) / decode_s,
        "tp": config.tp_size,
        "pp": config.pp_size,
        "device_ids": config.resolved_device_ids(),
        "eager": config.enforce_eager,
    }


def measure(
    config: EngineConfig,
    prompt: str | list[int],
    *,
    n: int = 64,
    warmup: int = 2,
    repeats: int = 3,
) -> dict[str, Any]:
    with engine_session(config) as eng:
        return measure_with_engine(
            eng,
            config,
            prompt,
            n=n,
            warmup=warmup,
            repeats=repeats,
        )


@lru_cache(maxsize=8)
def _tokenizer(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def decode_token_ids(
    model: str | os.PathLike[str],
    token_ids: list[int],
    *,
    skip_special_tokens: bool = True,
) -> str:
    return _tokenizer(str(model)).decode(
        token_ids,
        skip_special_tokens=skip_special_tokens,
    )
