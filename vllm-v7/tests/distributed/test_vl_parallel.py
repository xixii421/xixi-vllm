"""VL TP/PP + mm_encoder_tp_mode + CUDA Graph (needs >=2 GPUs + Qwen3-VL)."""

from __future__ import annotations

import pytest

from tests.conftest import needs_2gpu
from tests.helpers import VL_MODEL, cfg, greedy, require_model, solid_image

pytestmark = [pytest.mark.distributed, needs_2gpu, require_model(VL_MODEL)]


def _vl_base(**kwargs):
    return cfg(VL_MODEL, max_num_seqs=2, **kwargs)


@pytest.mark.parametrize("mode", ["data", "weights"])
def test_tp2_matches_single_and_syncs_mrope(mode: str):
    img = solid_image((64, 64), (32, 128, 200))
    prompt = "Describe the image briefly."
    base = dict(mm_encoder_tp_mode=mode)
    single = greedy(
        _vl_base(**base, tp_size=1, pp_size=1, device_ids=[1]),
        prompt,
        n=6,
        images=[img],
    )
    tp2 = greedy(
        _vl_base(**base, tp_size=2, pp_size=1, device_ids=[0, 1]),
        prompt,
        n=6,
        images=[img],
    )
    assert tp2["rope_delta"] == single["rope_delta"]
    assert tp2["mrope_next"] == single["mrope_next"]
    assert tp2["tokens"] == single["tokens"]


def test_tp2_multi_image_data_mode():
    imgs = [
        solid_image((96, 64), (200, 20, 20)),
        solid_image((48, 96), (20, 200, 20)),
    ]
    prompt = "Compare the two images in one short sentence."
    base = dict(mm_encoder_tp_mode="data", context_len=2048, num_kvcache_blocks=48)
    single = greedy(
        _vl_base(**base, tp_size=1, pp_size=1, device_ids=[1]),
        prompt,
        n=4,
        images=imgs,
    )
    tp2 = greedy(
        _vl_base(**base, tp_size=2, pp_size=1, device_ids=[0, 1]),
        prompt,
        n=4,
        images=imgs,
    )
    assert tp2["tokens"] == single["tokens"]
    assert tp2["rope_delta"] == single["rope_delta"]


def test_pp2_matches_single_prefix():
    """First PP stage (ViT + early layers) on 4090 to avoid 4070 OOM."""
    img = solid_image((64, 64), (200, 40, 40))
    prompt = "What color dominates the image?"
    single = greedy(
        _vl_base(tp_size=1, pp_size=1, device_ids=[1]),
        prompt,
        n=6,
        images=[img],
    )
    pp2 = greedy(
        _vl_base(tp_size=1, pp_size=2, device_ids=[1, 0]),
        prompt,
        n=6,
        images=[img],
    )
    assert pp2["rope_delta"] == single["rope_delta"]
    assert pp2["tokens"][:4] == single["tokens"][:4]


def test_tp2_cuda_graph_matches_eager():
    img = solid_image((48, 48), (10, 200, 10))
    prompt = "Say one word about the image."
    common = dict(
        tp_size=2,
        pp_size=1,
        device_ids=[0, 1],
        mm_encoder_tp_mode="data",
    )
    eager = greedy(
        _vl_base(**common, enforce_eager=True), prompt, n=4, images=[img]
    )
    cg = greedy(
        _vl_base(**common, enforce_eager=False), prompt, n=4, images=[img]
    )
    assert cg["tokens"] == eager["tokens"]
    assert cg["rope_delta"] == eager["rope_delta"]
