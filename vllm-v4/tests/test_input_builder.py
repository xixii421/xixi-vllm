"""Input builder: prefill / decode tensors and slot mapping."""

from __future__ import annotations

import pytest
import torch

from src.engine.input_builder import prepare_decode, prepare_prefill
from src.memory.sequence import Sequence
from src.sampling_params import SamplingParams


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _seq(tokens, *, scheduled=None, cached=0, blocks=None, bs=16):
    Sequence.block_size = bs
    s = Sequence(tokens, SamplingParams(max_tokens=8))
    s.num_cached_tokens = cached
    s.num_scheduled_tokens = scheduled if scheduled is not None else len(tokens) - cached
    s.block_table = list(blocks or [])
    return s


class TestInputBuilder:
    def test_prepare_prefill(self):
        dev = torch.device("cuda")
        seq = _seq([1, 2, 3, 4, 5], scheduled=5, blocks=[0])
        ids, pos, ctx = prepare_prefill([seq], 16, dev)
        assert ids.tolist() == [1, 2, 3, 4, 5]
        assert pos.tolist() == [0, 1, 2, 3, 4]
        assert ctx["cu_seqlens_q"].tolist() == [0, 5]
        assert ctx["slot_mapping"].tolist() == [0, 1, 2, 3, 4]

    def test_prepare_prefill_prefix_resume(self):
        dev = torch.device("cuda")
        # First 8 cached on block 0; schedule remaining 4 into block 1
        seq = _seq([1] * 12, cached=8, scheduled=4, blocks=[0, 1], bs=8)
        ids, pos, ctx = prepare_prefill([seq], 8, dev)
        assert ids.tolist() == [1, 1, 1, 1]
        assert pos.tolist() == [8, 9, 10, 11]
        # slots: block1 starts at 1*8
        assert ctx["slot_mapping"].tolist() == [8, 9, 10, 11]

    def test_prepare_decode(self):
        dev = torch.device("cuda")
        seq = _seq([10, 20, 30], blocks=[5], bs=16)
        # last_block_num_tokens=3 → slot = 5*16 + 2
        ids, pos, ctx = prepare_decode([seq], 16, dev)
        assert ids.tolist() == [30]
        assert pos.tolist() == [2]
        assert ctx["context_lens"].tolist() == [3]
        assert ctx["slot_mapping"].tolist() == [5 * 16 + 2]

    def test_multiseq_prefill_cu(self):
        dev = torch.device("cuda")
        s1 = _seq([1, 2, 3], scheduled=3, blocks=[0])
        s2 = _seq([4, 5], scheduled=2, blocks=[1])
        ids, pos, ctx = prepare_prefill([s1, s2], 16, dev)
        assert ctx["cu_seqlens_q"].tolist() == [0, 3, 5]
        assert ids.tolist() == [1, 2, 3, 4, 5]
