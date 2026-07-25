"""Sequence + slot helpers."""

from __future__ import annotations

import pytest

from src.scheduling.sequence import Sequence, SequenceStatus, slot_of, slots_for_range
from src.sampling import SamplingParams
from src.runtime.execute.request import DecodeSequenceMetadata, compact_execute_sequences


class TestSequence:
    def test_lengths(self):
        s = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=4), block_size=16)
        assert len(s) == 4
        assert s.num_prompt_tokens == 4
        assert s.num_completion_tokens == 0
        s.append_token(9)
        assert s.num_tokens == 5 and s.last_token == 9
        assert s.num_completion_tokens == 1

    def test_num_blocks_and_last(self):
        s = Sequence([1] * 10, block_size=4)
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

    def test_decode_rpc_metadata_is_compact(self):
        import pickle

        s = Sequence(
            list(range(1024)),
            SamplingParams(temperature=0.0, max_tokens=16),
            block_size=16,
        )
        s.block_table = list(range(64))
        s.num_cached_tokens = 1024
        s.num_scheduled_tokens = 1

        wire = compact_execute_sequences([s], is_prefill=False)
        assert isinstance(wire[0], DecodeSequenceMetadata)
        assert len(wire[0]) == len(s)
        assert wire[0].last_token == s.last_token
        assert wire[0].last_block_num_tokens == s.last_block_num_tokens
        assert len(pickle.dumps(wire)) < len(pickle.dumps([s])) // 2

    def test_prefill_rpc_uses_wire_without_aliasing_engine_seq(self):
        from src.runtime.execute.request import PrefillSequenceWire

        s = Sequence([1, 2, 3])
        wire = compact_execute_sequences([s], is_prefill=True)
        assert isinstance(wire[0], PrefillSequenceWire)
        assert wire[0] is not s
        assert wire[0].token_ids == s.token_ids
        assert wire[0].seq_id == s.seq_id
