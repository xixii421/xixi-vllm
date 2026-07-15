"""Unit tests for TP/PP math helpers (no multi-GPU required)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.distributed.pp import get_pp_indices
from src.distributed.math import divide, ensure_divisibility


def test_ensure_divisibility_ok():
    ensure_divisibility(8, 2)
    ensure_divisibility(0, 1)


def test_ensure_divisibility_raises():
    try:
        ensure_divisibility(7, 2)
        assert False, "expected AssertionError"
    except AssertionError:
        pass


def test_divide():
    assert divide(16, 4) == 4
    assert divide(8, 8) == 1


def test_get_pp_indices_even():
    assert get_pp_indices(28, 0, 2) == (0, 14)
    assert get_pp_indices(28, 1, 2) == (14, 28)


def test_get_pp_indices_remainder_on_last():
    # 7 layers, 3 PP stages → 2+2+3
    assert get_pp_indices(7, 0, 3) == (0, 2)
    assert get_pp_indices(7, 1, 3) == (2, 4)
    assert get_pp_indices(7, 2, 3) == (4, 7)


def test_get_pp_indices_pp1():
    assert get_pp_indices(12, 0, 1) == (0, 12)


def test_column_parallel_shard_math():
    """Column-parallel weight shard: rank r gets rows [r*S:(r+1)*S]."""
    out, inn, tp = 16, 8, 4
    full = torch.randn(out, inn)
    shard = out // tp
    for rank in range(tp):
        piece = full.narrow(0, rank * shard, shard)
        assert piece.shape == (shard, inn)
        # F.linear(x, piece) produces local Y_i
        x = torch.randn(2, inn)
        y_local = F.linear(x, piece)
        assert y_local.shape == (2, shard)
