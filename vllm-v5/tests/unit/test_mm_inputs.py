"""Multimodal contracts: hasher, spans, truncate."""

from __future__ import annotations

import torch

from src.runtime.input.hasher import hash_image_pixels
from src.runtime.input.features import (
    ImageSpanInfo,
    build_mm_features,
    overlaps_window,
    span_end,
    truncate_mm_features,
)
from src.sampling import SamplingParams
from src.scheduling.sequence import MultiModalFeature, Sequence


def test_hash_stable_and_content_sensitive():
    a = torch.zeros(4, 8, dtype=torch.float16)
    b = torch.zeros(4, 8, dtype=torch.float16)
    c = torch.ones(4, 8, dtype=torch.float16)
    assert hash_image_pixels(a) == hash_image_pixels(b)
    assert hash_image_pixels(a) != hash_image_pixels(c)


def test_build_mm_features_slices_and_hashes():
    # two images, 2 and 3 patches
    pv = torch.arange(5 * 4, dtype=torch.float32).view(5, 4)
    spans = [
        ImageSpanInfo(
            start=1,
            end=5,
            num_encoder_tokens=4,
            grid_thw=torch.tensor([[1, 2, 1]]),
        ),
        ImageSpanInfo(
            start=6,
            end=10,
            num_encoder_tokens=4,
            grid_thw=torch.tensor([[1, 3, 1]]),
        ),
    ]
    feats = build_mm_features(pv, spans)
    assert len(feats) == 2
    assert feats[0].data.shape[0] == 2
    assert feats[1].data.shape[0] == 3
    assert feats[0].uid != feats[1].uid
    assert span_end(feats[0]) == 5


def test_truncate_keeps_fully_contained_spans():
    mf = MultiModalFeature(
        modality="image",
        data=torch.zeros(2, 2),
        start=12,
        end=16,
        num_encoder_tokens=4,
        uid="u",
        grid_thw=None,
    )
    tokens = list(range(20))
    kept_tok, kept_mm = truncate_mm_features(tokens, [mf], context_len=10)
    # keep_from=10 → image [12,16) fully inside kept window [10,20)
    assert kept_tok == list(range(10, 20))
    assert len(kept_mm) == 1
    assert kept_mm[0].start == 2
    assert kept_mm[0].end == 6


def test_overlaps_window():
    mf = MultiModalFeature(
        modality="image",
        data=torch.zeros(1),
        start=10,
        end=20,
        num_encoder_tokens=10,
        uid="u",
    )
    assert overlaps_window(mf, 0, 10) is False
    assert overlaps_window(mf, 5, 15) is True
    seq = Sequence([1] * 30, SamplingParams(max_tokens=2), mm_features=[mf])
    assert seq.has_multimodal
