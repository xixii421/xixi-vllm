"""v4 引擎测量：单副本 TTFT/decode/batch，或多副本聚合吞吐。"""

from __future__ import annotations

import time

import torch

from common import (
    BATCH_SIZE,
    DP_N_REQS,
    DP_PROMPT_LEN,
    MAX_TOKENS,
    PROMPT_LEN,
    dummy_ids,
)


def warmup_single(engine, *, rounds: int = 3) -> None:
    from src import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
    for it in range(rounds):
        seq = engine.add_request(dummy_ids(PROMPT_LEN, salt=it), sp)
        while not seq.is_finished:
            engine.step()
    torch.cuda.synchronize()


def measure_ttft_decode(engine) -> tuple[float, float]:
    """返回 (ttft_ms, decode_tps)。"""
    from src import SamplingParams

    seq = engine.add_request(
        dummy_ids(PROMPT_LEN, salt=100),
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=MAX_TOKENS),
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ttft = None
    while not seq.is_finished:
        engine.step()
        if ttft is None and seq.num_completion_tokens > 0:
            torch.cuda.synchronize()
            ttft = time.perf_counter() - t0
    torch.cuda.synchronize()
    e2e = time.perf_counter() - t0
    assert ttft is not None
    n_out = seq.num_completion_tokens
    decode_tps = (n_out - 1) / (e2e - ttft) if e2e > ttft else 0.0
    return ttft * 1000.0, decode_tps


def measure_batch(engine, batch: int = BATCH_SIZE) -> float:
    """返回 batch 聚合吞吐 tok/s。"""
    from src import SamplingParams

    sps = [
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=MAX_TOKENS)
        for _ in range(batch)
    ]
    seqs = [
        engine.add_request(dummy_ids(PROMPT_LEN, salt=1000 + i), sps[i])
        for i in range(batch)
    ]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while engine.has_unfinished():
        engine.step()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    n = sum(s.num_completion_tokens for s in seqs)
    return n / wall if wall > 0 else 0.0


def warmup_dp(engine, n_reqs: int, *, rounds: int = 2) -> None:
    from src import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
    for it in range(rounds):
        for i in range(n_reqs):
            engine.add_request(dummy_ids(DP_PROMPT_LEN, salt=it * 13 + i), sp)
        while engine.has_unfinished():
            engine.step()
    torch.cuda.synchronize()


def measure_dp_throughput(engine, n_reqs: int = DP_N_REQS) -> tuple[float, int]:
    """返回 (聚合 tok/s, 总生成 token 数)。"""
    from src import SamplingParams

    sp = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=MAX_TOKENS)
    seqs = [
        engine.add_request(dummy_ids(DP_PROMPT_LEN, salt=1000 + i), sp)
        for i in range(n_reqs)
    ]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while engine.has_unfinished():
        engine.step()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    n_tok = sum(s.num_completion_tokens for s in seqs)
    tps = n_tok / wall if wall > 0 else 0.0
    return tps, n_tok
