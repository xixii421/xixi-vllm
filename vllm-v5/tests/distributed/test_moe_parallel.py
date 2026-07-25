"""MoE TP / PP correctness on a tiny checkpoint (cutlass or Triton).

``moe_fused_backend=auto``：有 cutlass 用 cutlass，否则 Triton。
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import needs_2gpu, needs_4gpu
from tests.helpers import cfg, greedy_tokens, homo_device_ids
from tests.helpers.devices import blackwell_a100_pair, env_device_ids, gpu_name
from tests.helpers.moe_tiny import build_tiny_moe
from tests.helpers.parallel import layout_ok

pytestmark = [pytest.mark.distributed, needs_2gpu]

PROMPT = [1, 2, 3, 4, 5, 6, 7, 8]
N = 8


@pytest.fixture(scope="module")
def tiny_moe_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_moe_dist")
    return build_tiny_moe(d)


def _moe_cfg(tiny_moe_dir, **kwargs):
    base = dict(
        max_num_seqs=1,
        context_len=256,
        num_kvcache_blocks=16,
        gpu_memory_utilization=0.35,
        enforce_eager=True,
        enable_radix=False,
        dtype="float16",
        moe_fused_backend="auto",
    )
    base.update(kwargs)
    return cfg(tiny_moe_dir, **base)


def _hetero_pair() -> list[int]:
    override = env_device_ids()
    if override is not None and len(override) == 2:
        return override
    pair = blackwell_a100_pair(min_free_mib=0)
    if pair is None:
        pytest.skip("need Blackwell + A100 (or VLLM_V7_MOE_DEVICES=0,3)")
    return pair


def test_tp2_matches_single(tiny_moe_dir):
    ok, reason = layout_ok(tiny_moe_dir, tp=2)
    if not ok:
        pytest.skip(reason)
    ids = homo_device_ids(2)
    single = greedy_tokens(_moe_cfg(tiny_moe_dir, device_ids=[ids[0]]), PROMPT, n=N)
    tp2 = greedy_tokens(
        _moe_cfg(tiny_moe_dir, tp_size=2, device_ids=ids),
        PROMPT,
        n=N,
    )
    assert tp2 == single


def test_pp2_matches_single(tiny_moe_dir):
    ok, reason = layout_ok(tiny_moe_dir, tp=1, pp=2)
    if not ok:
        pytest.skip(reason)
    ids = homo_device_ids(2)
    single = greedy_tokens(_moe_cfg(tiny_moe_dir, device_ids=[ids[0]]), PROMPT, n=N)
    pp2 = greedy_tokens(
        _moe_cfg(tiny_moe_dir, tp_size=1, pp_size=2, device_ids=ids),
        PROMPT,
        n=N,
    )
    assert pp2 == single


def test_pp2_hetero_bw_a100_matches_single(tiny_moe_dir):
    ok, reason = layout_ok(tiny_moe_dir, tp=1, pp=2)
    if not ok:
        pytest.skip(reason)
    pair = _hetero_pair()
    print(f"[tiny] PP2 hetero {pair} = {gpu_name(pair[0])} | {gpu_name(pair[1])}")
    single = greedy_tokens(_moe_cfg(tiny_moe_dir, device_ids=[pair[0]]), PROMPT, n=N)
    pp2 = greedy_tokens(
        _moe_cfg(tiny_moe_dir, tp_size=1, pp_size=2, device_ids=pair),
        PROMPT,
        n=N,
    )
    assert pp2 == single


def test_triton_backend_tp1(tiny_moe_dir):
    gid = homo_device_ids(2)[0]
    toks = greedy_tokens(
        _moe_cfg(tiny_moe_dir, device_ids=[gid], moe_fused_backend="triton"),
        PROMPT,
        n=N,
    )
    assert len(toks) == N


@needs_4gpu
def test_tp2pp2_matches_single_prefix(tiny_moe_dir):
    if torch.cuda.device_count() < 4:
        pytest.skip("need 4 GPUs for tp2pp2")
    ok, reason = layout_ok(tiny_moe_dir, tp=2, pp=2)
    if not ok:
        pytest.skip(reason)
    ids = homo_device_ids(4)
    single = greedy_tokens(_moe_cfg(tiny_moe_dir, device_ids=[ids[0]]), PROMPT, n=N)
    tp2pp2 = greedy_tokens(
        _moe_cfg(tiny_moe_dir, tp_size=2, pp_size=2, device_ids=ids),
        PROMPT,
        n=N,
    )
    assert tp2pp2[:4] == single[:4]
