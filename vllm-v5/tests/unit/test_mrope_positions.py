"""Unit tests: v5 mRoPE positions aligned with v4 / HF image formula."""

from __future__ import annotations

import numpy as np
import torch

from src.runtime.vision.positions import (
    compute_mrope_positions,
    mrope_positions,
)
from src.sampling import SamplingParams
from src.scheduling import MultiModalFeature, Sequence


def _feat(start: int, n: int, grid: tuple[int, int, int]) -> MultiModalFeature:
    return MultiModalFeature(
        modality="image",
        data=torch.zeros(1),
        start=start,
        end=start + n,
        num_encoder_tokens=n,
        uid=f"img@{start}",
        grid_thw=torch.tensor([list(grid)], dtype=torch.int64),
        grid_thw_tuple=grid,
    )


def _seq(token_ids: list[int], feats: list[MultiModalFeature]) -> Sequence:
    return Sequence(token_ids, SamplingParams(max_tokens=4), mm_features=feats)


def _hf_vision(start: int, grid: tuple[int, int, int], merge: int) -> np.ndarray:
    t, h, w = grid
    gh, gw = h // merge, w // merge
    pt = np.arange(t, dtype=np.int64)
    ph = np.arange(gh, dtype=np.int64) + start
    pw = np.arange(gw, dtype=np.int64) + start
    T, H, W = np.meshgrid(pt, ph, pw, indexing="ij")
    out = np.stack([T, H, W], axis=0).reshape(3, -1)
    out[0] += start
    return out


def test_text_only():
    seq = _seq(list(range(5)), [])
    seq.num_cached_tokens = 0
    seq.num_scheduled_tokens = 5
    pos = mrope_positions(torch.arange(5), [seq], 2, torch.device("cpu"))
    assert pos.shape == (3, 5)
    for a in range(3):
        assert pos[a].tolist() == [0, 1, 2, 3, 4]
    assert seq.rope_delta == 0
    assert seq.mrope_next == 5


def test_single_image_matches_hf():
    merge = 2
    grid = (1, 4, 4)
    n_vis = 4
    seq = _seq(list(range(7)), [_feat(2, n_vis, grid)])
    seq.num_cached_tokens = 0
    seq.num_scheduled_tokens = 7
    pos = mrope_positions(torch.arange(7), [seq], merge, torch.device("cpu"))
    assert np.array_equal(pos[:, :2].numpy(), np.arange(2).reshape(1, -1).repeat(3, 0))
    assert np.array_equal(pos[:, 2:6].numpy(), _hf_vision(2, grid, merge))
    assert seq.mm_features[0].mrope_origin == 2
    assert seq.mrope_next == int(pos.max().item()) + 1
    assert (len(seq) - 1) + seq.rope_delta == seq.mrope_next - 1


def test_compute_does_not_mutate_until_apply():
    seq = _seq(list(range(7)), [_feat(2, 4, (1, 4, 4))])
    seq.num_cached_tokens = 0
    seq.num_scheduled_tokens = 7
    assert seq.mrope_next is None
    pos, updates = compute_mrope_positions(
        torch.arange(7), [seq], 2, torch.device("cpu")
    )
    assert seq.mrope_next is None
    assert seq.mm_features[0].mrope_origin is None
    assert len(updates) == 1
    from src.runtime.vision.positions import apply_mrope_updates

    apply_mrope_updates([seq], updates)
    assert seq.mrope_next is not None
    assert seq.mm_features[0].mrope_origin == 2
    assert pos.shape == (3, 7)


def test_chunked_equals_full():
    merge = 2
    grid = (1, 8, 8)
    n_vis = 16
    seq_len = 3 + n_vis + 5

    def run(chunk: int) -> tuple[np.ndarray, int, int]:
        seq = _seq(list(range(seq_len)), [_feat(3, n_vis, grid)])
        parts = []
        cached = 0
        while cached < seq_len:
            m = min(chunk, seq_len - cached)
            seq.num_cached_tokens = cached
            seq.num_scheduled_tokens = m
            parts.append(
                mrope_positions(
                    torch.arange(m), [seq], merge, torch.device("cpu")
                ).numpy()
            )
            cached += m
        return np.concatenate(parts, axis=1), seq.rope_delta, int(seq.mrope_next)

    full, d0, n0 = run(seq_len)
    for cs in (1, 4, 7, 16):
        pos, d, n = run(cs)
        assert np.array_equal(full, pos), cs
        assert d == d0 and n == n0


def test_grid_thw_tuple_without_tensor():
    feat = MultiModalFeature(
        modality="image",
        data=None,
        start=1,
        end=5,
        num_encoder_tokens=4,
        uid="t",
        grid_thw=None,
        grid_thw_tuple=(1, 4, 4),
    )
    seq = _seq(list(range(6)), [feat])
    seq.num_cached_tokens = 0
    seq.num_scheduled_tokens = 6
    pos = mrope_positions(torch.arange(6), [seq], 2, torch.device("cpu"))
    assert pos.shape == (3, 6)
    assert seq.mm_features[0].mrope_origin == 1
