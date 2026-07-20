"""Unit tests for distributed helpers (no multi-GPU required)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.distributed.math import divide, ensure_divisibility
from src.distributed.pp import IntermediateTensors, get_layer_id, get_pp_indices
from src.distributed.parallel_state import RankGroup


def test_ensure_divisibility_ok():
    ensure_divisibility(8, 2)
    ensure_divisibility(0, 1)


def test_ensure_divisibility_raises():
    with pytest.raises(ValueError, match="not divisible"):
        ensure_divisibility(7, 2)


def test_divide():
    assert divide(16, 4) == 4
    assert divide(8, 8) == 1


def test_get_pp_indices_even():
    assert get_pp_indices(28, 0, 2) == (0, 14)
    assert get_pp_indices(28, 1, 2) == (14, 28)


def test_get_pp_indices_balanced_remainder():
    # 7 layers, 3 stages → 3+2+2 (first ranks absorb remainder)
    assert get_pp_indices(7, 0, 3) == (0, 3)
    assert get_pp_indices(7, 1, 3) == (3, 5)
    assert get_pp_indices(7, 2, 3) == (5, 7)


def test_get_pp_indices_pp1():
    assert get_pp_indices(12, 0, 1) == (0, 12)


def test_get_pp_indices_five_over_two():
    assert get_pp_indices(5, 0, 2) == (0, 3)
    assert get_pp_indices(5, 1, 2) == (3, 5)


def test_rank_group_neighbors():
    g = RankGroup([0, 2, 4], rank=2)
    assert g.group_rank == 1
    assert g.prev_rank == 0
    assert g.next_rank == 4
    assert not g.is_first_rank
    assert not g.is_last_rank


def test_intermediate_tensors_dataclass():
    h = torch.zeros(2, 4)
    r = torch.ones(2, 4)
    t = IntermediateTensors(hidden_states=h, residual=r)
    assert t.hidden_states is h
    assert t.residual is r


def test_get_layer_id():
    assert get_layer_id("model.layers.12.self_attn.q_proj.weight") == 12
    assert get_layer_id("lm_head.weight") is None


def test_column_parallel_shard_math():
    out, inn, tp = 16, 8, 4
    full = torch.randn(out, inn)
    shard = out // tp
    for rank in range(tp):
        piece = full.narrow(0, rank * shard, shard)
        assert piece.shape == (shard, inn)
        x = torch.randn(2, inn)
        y_local = F.linear(x, piece)
        assert y_local.shape == (2, shard)
