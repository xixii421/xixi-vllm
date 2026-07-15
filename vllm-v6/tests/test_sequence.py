"""Sequence + slot helpers."""

from __future__ import annotations

import pytest

from src.scheduling.sequence import Sequence, SequenceStatus, slot_of, slots_for_range
from src.sampling_params import SamplingParams


class TestSequence:
    def test_lengths(self):
        Sequence.block_size = 16
        s = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=4))
        assert len(s) == 4
        assert s.num_prompt_tokens == 4
        assert s.num_completion_tokens == 0
        s.append_token(9)
        assert s.num_tokens == 5 and s.last_token == 9
        assert s.num_completion_tokens == 1

    def test_num_blocks_and_last(self):
        Sequence.block_size = 4
        s = Sequence([1] * 10)
        assert s.num_blocks == 3
        assert s.last_block_num_tokens == 2
        assert s.block_token_ids(1) == [1] * 4

    def test_slot_of(self):
        assert slot_of([7, 8], 0, 16) == 7 * 16
        assert slot_of([7, 8], 16, 16) == 8 * 16
        assert slot_of([7, 8], 17, 16) == 8 * 16 + 1
        with pytest.raises(IndexError):
            slot_of([7], 16, 16)

    def test_slots_for_range(self):
        slots = slots_for_range([3], 0, 3, 16)
        assert slots == [48, 49, 50]

    def test_sampling_params_reject_negative_max(self):
        with pytest.raises(ValueError):
            SamplingParams(max_tokens=-1)
