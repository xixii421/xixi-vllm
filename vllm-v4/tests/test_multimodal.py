"""CPU-only unit tests for v4 multimodal control logic.

Covers the two pieces that must be correct regardless of GPU/model:
* MM-aware APC cap (`Sequence.prefix_cacheable_full_blocks`)
* MRoPE 3-axis position builder + decode `rope_delta` continuation.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.sequence import MultiModalFeature, Sequence  # noqa: E402
from src.multimodal._mrope import mrope_positions  # noqa: E402


def _img_feature(start: int, n: int, grid_thw) -> MultiModalFeature:
    return MultiModalFeature(
        modality="image",
        data=torch.zeros(1),
        start=start,
        end=start + n,
        num_encoder_tokens=n,
        uid=f"img@{start}",
        grid_thw=torch.tensor(grid_thw, dtype=torch.int64),
    )


def test_prefix_cap_text_only():
    seq = Sequence(token_ids=list(range(20)), block_size=4)
    assert seq.num_full_blocks() == 5
    assert seq.prefix_cacheable_full_blocks() == 5  # no images → cache everything


def test_prefix_cap_with_image():
    # Image placeholder starts at token 6 → only block 0 (tokens 0..3) is cacheable.
    seq = Sequence(
        token_ids=list(range(20)),
        block_size=4,
        mm_features=[_img_feature(start=6, n=4, grid_thw=[[1, 4, 4]])],
    )
    assert seq.first_mm_token == 6
    assert seq.prefix_cacheable_full_blocks() == 1
    assert seq.has_multimodal


def test_reset_mrope_state():
    seq = Sequence(
        token_ids=list(range(8)),
        block_size=4,
        mm_features=[_img_feature(start=2, n=4, grid_thw=[[1, 4, 4]])],
    )
    seq.rope_delta = 7
    seq.mrope_next = 42
    seq.mm_features[0].mrope_origin = 3
    seq.reset_mrope_state()
    assert seq.rope_delta == 0
    assert seq.mrope_next is None
    assert seq.mm_features[0].mrope_origin is None


def test_mrope_text_only():
    seq = Sequence(token_ids=list(range(5)), block_size=8)
    seq.num_cached_tokens = 0
    seq.num_scheduled_tokens = 5
    ids = torch.arange(5)
    pos = mrope_positions(ids, [seq], merge_size=2, device=torch.device("cpu"))
    assert pos.shape == (3, 5)
    # Text tokens advance all three axes identically 0..4.
    for axis in range(3):
        assert pos[axis].tolist() == [0, 1, 2, 3, 4]
    assert seq.rope_delta == 0
    assert seq.mrope_next == 5


def test_mrope_with_image_advances_cursor():
    # layout: [t0, t1, img(2x2=4 merged tokens), t6]  → 7 tokens, merge_size=2, grid 1x4x4
    seq = Sequence(
        token_ids=list(range(7)),
        block_size=8,
        mm_features=[_img_feature(start=2, n=4, grid_thw=[[1, 4, 4]])],
    )
    seq.num_cached_tokens = 0
    seq.num_scheduled_tokens = 7
    ids = torch.arange(7)
    pos = mrope_positions(ids, [seq], merge_size=2, device=torch.device("cpu"))
    assert pos.shape == (3, 7)
    # First two text tokens: 0,1 on every axis.
    assert pos[0, 0].item() == 0 and pos[0, 1].item() == 1
    # Image origin sits at position 2; image spans 4 merged tokens (2x2 grid).
    assert seq.mm_features[0].mrope_origin == 2
    # A 2x2-merged image occupies only 2 position units for 4 tokens, so the
    # absolute cursor lags the token count → rope_delta is negative here.
    assert seq.mrope_next == int(pos.max().item()) + 1
    assert seq.rope_delta == seq.mrope_next - 1 - (len(seq) - 1)
    # Decode position must equal the next absolute cursor.
    assert (len(seq) - 1) + seq.rope_delta == seq.mrope_next - 1
