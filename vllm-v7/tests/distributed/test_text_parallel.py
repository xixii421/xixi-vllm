"""Text TP/PP: greedy match vs single-GPU + CUDA Graph parity (needs >=2 GPUs)."""

from __future__ import annotations

import pytest

from tests.conftest import needs_2gpu
from tests.helpers import (
    TEXT_MODEL,
    cfg,
    greedy_tokens,
    homo_device_ids,
    require_model,
    tp_divisible,
)

pytestmark = [pytest.mark.distributed, needs_2gpu, require_model(TEXT_MODEL)]

PROMPTS = ["The capital of France is", "Hello, my name is"]


def _batch(*, tp: int, pp: int, device_ids: list[int], n: int, eager: bool = True):
    conf = cfg(
        TEXT_MODEL,
        max_num_seqs=4,
        context_len=512,
        tp_size=tp,
        pp_size=pp,
        device_ids=device_ids,
        enforce_eager=eager,
    )
    return [greedy_tokens(conf, p, n=n) for p in PROMPTS]


def test_tp2_matches_single_gpu():
    ok, reason = tp_divisible(TEXT_MODEL, 2)
    if not ok:
        pytest.skip(reason)
    ids = homo_device_ids(2)
    n = 8
    single = _batch(tp=1, pp=1, device_ids=[ids[0]], n=n)
    tp2 = _batch(tp=2, pp=1, device_ids=ids, n=n)
    assert tp2 == single


def test_pp2_matches_single_gpu_prefix():
    # PP can use two different SKUs; pick first two visible devices.
    import torch

    if torch.cuda.device_count() < 2:
        pytest.skip("need 2 GPUs")
    n = 8
    single = _batch(tp=1, pp=1, device_ids=[0], n=n)
    pp2 = _batch(tp=1, pp=2, device_ids=[0, 1], n=n)
    for i, (a, b) in enumerate(zip(single, pp2)):
        assert a[:4] == b[:4], f"prompt {i}: single={a!r} pp2={b!r}"


def test_tp2_cuda_graph_matches_eager():
    ok, reason = tp_divisible(TEXT_MODEL, 2)
    if not ok:
        pytest.skip(reason)
    ids = homo_device_ids(2)
    prompt = PROMPTS[0]
    common = dict(max_num_seqs=4, context_len=512, tp_size=2, pp_size=1, device_ids=ids)
    eager = greedy_tokens(cfg(TEXT_MODEL, **common, enforce_eager=True), prompt, n=8)
    cg = greedy_tokens(cfg(TEXT_MODEL, **common, enforce_eager=False), prompt, n=8)
    assert cg == eager


@pytest.mark.parametrize("tp", [2, 3, 4])
def test_tp_size_supported_or_skip(tp: int):
    """Document TP eligibility; actual match covered by bench_tp / tp2 tests."""
    ok, reason = tp_divisible(TEXT_MODEL, tp)
    if not ok:
        pytest.skip(reason)
    ids = homo_device_ids(tp)
    assert len(ids) == tp
