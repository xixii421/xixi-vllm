"""Qwen3 / Qwen2 MoE (real weights): cutlass or Triton; prefer smaller A2.7B.

Parallel policy
---------------
- **TP** → same SKU (default ``allow_hetero_tp=False``).
- **PP** → stages may differ SKU.
- Backend: ``moe_fused_backend=auto`` → cutlass if SM+toolkit OK else Triton.

Env
---
- ``VLLM_V7_MOE_DEVICES=0,3``  force PP pair
"""

from __future__ import annotations

import time

import pytest
import torch

from src.config.moe_backend import resolve_moe_backend
from tests.conftest import needs_2gpu, needs_cuda
from tests.helpers import MOE_MODEL, cfg, greedy_tokens, homo_device_ids, measure, require_model
from tests.helpers.devices import (
    blackwell_a100_pair,
    env_device_ids,
    find_a100,
    find_blackwell,
    gpu_free_mib,
    gpu_name,
)
from tests.helpers.parallel import layout_ok

pytestmark = [pytest.mark.gpu, needs_cuda, require_model(MOE_MODEL)]

PROMPT = "The capital of France is"
N_CORRECT = 8
N_PERF = 32


def _any_gpu(min_free_mib: int = 0) -> int:
    for fn in (find_blackwell, find_a100):
        g = fn(min_free_mib=min_free_mib)
        if g is not None:
            return g
    return 0


def _pair_for_pp(min_free_mib: int = 0) -> list[int]:
    override = env_device_ids()
    if override is not None and len(override) == 2:
        return override
    pair = blackwell_a100_pair(min_free_mib=min_free_mib)
    if pair is not None:
        return pair
    try:
        return homo_device_ids(2)
    except Exception:
        pytest.skip(
            "need a 2-GPU pair (or set VLLM_V7_MOE_DEVICES=a,b); "
            f"free={[gpu_free_mib(i) for i in range(torch.cuda.device_count())]}"
        )


def _base(**kwargs):
    d = dict(
        max_num_seqs=1,
        context_len=512,
        num_kvcache_blocks=16,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        enable_radix=False,
        moe_fused_backend="auto",
    )
    d.update(kwargs)
    return cfg(MOE_MODEL, **d)


def test_moe_tp1_greedy():
    gid = _any_gpu()
    backend = resolve_moe_backend("auto", device=gid)
    print(
        f"[moe] TP1 device={gid} ({gpu_name(gid)}) backend={backend} "
        f"model={MOE_MODEL} free={gpu_free_mib(gid)}MiB"
    )
    toks = greedy_tokens(_base(device_ids=[gid]), PROMPT, n=N_CORRECT)
    assert len(toks) == N_CORRECT


@needs_2gpu
def test_moe_pp2_matches_single():
    ok, reason = layout_ok(MOE_MODEL, tp=1, pp=2)
    if not ok:
        pytest.skip(reason)
    pair = _pair_for_pp()
    print(f"[moe] PP2 devices={pair} ({gpu_name(pair[0])} | {gpu_name(pair[1])})")
    single = greedy_tokens(_base(device_ids=[pair[0]]), PROMPT, n=N_CORRECT)
    pp2 = greedy_tokens(
        _base(tp_size=1, pp_size=2, device_ids=pair),
        PROMPT,
        n=N_CORRECT,
    )
    assert pp2 == single


@needs_2gpu
def test_moe_perf_tp1_vs_pp2():
    ok, reason = layout_ok(MOE_MODEL, tp=1, pp=2)
    if not ok:
        pytest.skip(reason)
    pair = _pair_for_pp()
    tp1_id = pair[0]

    print(f"[moe-perf] TP1 on {tp1_id} ({gpu_name(tp1_id)}) …")
    t0 = time.perf_counter()
    m_tp1 = measure(
        _base(device_ids=[tp1_id], num_kvcache_blocks=24),
        PROMPT,
        n=N_PERF,
        warmup=0,
        repeats=1,
    )
    dt_tp1 = time.perf_counter() - t0

    print(f"[moe-perf] PP2 {pair} …")
    t1 = time.perf_counter()
    m_pp2 = measure(
        _base(tp_size=1, pp_size=2, device_ids=pair, num_kvcache_blocks=24),
        PROMPT,
        n=N_PERF,
        warmup=0,
        repeats=1,
    )
    dt_pp2 = time.perf_counter() - t1

    print(
        f"[moe-perf] TP1  decode={m_tp1['decode_tok_s']:.2f} tok/s "
        f"ttft={m_tp1.get('ttft_ms')} wall={dt_tp1:.1f}s"
    )
    print(
        f"[moe-perf] PP2  decode={m_pp2['decode_tok_s']:.2f} tok/s "
        f"ttft={m_pp2.get('ttft_ms')} wall={dt_pp2:.1f}s"
    )
    assert len(m_tp1["tokens"]) == N_PERF
    assert len(m_pp2["tokens"]) == N_PERF
