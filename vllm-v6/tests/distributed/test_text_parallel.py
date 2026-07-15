"""Text TP/PP: greedy match vs single-GPU + CUDA Graph parity (needs >=2 GPUs).

Heterogeneous GPUs (e.g. 4070+4090): PP may drift after a few decode steps;
we assert a short greedy prefix. TP all-reduces each layer and stays matched longer.
"""

from __future__ import annotations

import pytest

from tests.conftest import needs_2gpu
from tests.helpers import TEXT_MODEL, cfg, greedy_tokens, require_model

pytestmark = [pytest.mark.distributed, needs_2gpu, require_model(TEXT_MODEL)]

PROMPTS = ["The capital of France is", "Hello, my name is"]


def _batch_tokens(*, tp: int, pp: int, device_ids: list[int], n: int) -> list[list[int]]:
    conf = cfg(
        TEXT_MODEL,
        max_num_seqs=4,
        context_len=512,
        tp_size=tp,
        pp_size=pp,
        device_ids=device_ids,
    )
    return [greedy_tokens(conf, p, n=n) for p in PROMPTS]


def test_tp2_matches_single_gpu():
    n = 8
    single = _batch_tokens(tp=1, pp=1, device_ids=[0], n=n)
    tp2 = _batch_tokens(tp=2, pp=1, device_ids=[0, 1], n=n)
    assert tp2 == single


def test_pp2_matches_single_gpu_prefix():
    n = 8
    single = _batch_tokens(tp=1, pp=1, device_ids=[0], n=n)
    pp2 = _batch_tokens(tp=1, pp=2, device_ids=[0, 1], n=n)
    for i, (a, b) in enumerate(zip(single, pp2)):
        assert a[:4] == b[:4], f"prompt {i}: single={a!r} pp2={b!r}"


def test_tp2_cuda_graph_matches_eager():
    prompt = PROMPTS[0]
    common = dict(
        max_num_seqs=4,
        context_len=512,
        tp_size=2,
        pp_size=1,
        device_ids=[0, 1],
    )
    eager = greedy_tokens(cfg(TEXT_MODEL, **common, enforce_eager=True), prompt, n=8)
    cg = greedy_tokens(cfg(TEXT_MODEL, **common, enforce_eager=False), prompt, n=8)
    assert cg == eager


def test_pp2_cuda_graph_prefix_matches_eager():
    prompt = PROMPTS[1]
    common = dict(
        max_num_seqs=4,
        context_len=512,
        tp_size=1,
        pp_size=2,
        device_ids=[0, 1],
    )
    eager = greedy_tokens(cfg(TEXT_MODEL, **common, enforce_eager=True), prompt, n=6)
    cg = greedy_tokens(cfg(TEXT_MODEL, **common, enforce_eager=False), prompt, n=6)
    assert cg[:4] == eager[:4]
