"""Unit tests for ViT batch DP (mm_encoder_tp_mode=data) helpers."""

from __future__ import annotations

from src.multimodal.vision_dp import get_load_balance_assignment


def test_load_balance_empty():
    shuffle, counts, loads = get_load_balance_assignment([], 2)
    assert shuffle == []
    assert counts == [0, 0]
    assert loads == [0, 0]


def test_load_balance_greedy_by_size():
    # Larger patches go first; pack onto lighter GPU.
    shuffle, counts, loads = get_load_balance_assignment([10, 1, 9, 2], 2)
    assert sorted(shuffle) == [0, 1, 2, 3]
    assert sum(counts) == 4
    assert sum(loads) == 22
    # Ideal split ~11 / 11
    assert abs(loads[0] - loads[1]) <= 2


def test_load_balance_single_gpu():
    shuffle, counts, loads = get_load_balance_assignment([3, 5], 1)
    assert shuffle == [1, 0] or shuffle == [0, 1]
    assert counts == [2]
    assert loads == [8]
