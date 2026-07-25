"""Scheduler: prefill / decode / chunk / eos / preempt / max_tokens=0 / prefix promote."""

from __future__ import annotations

from src.scheduling import Scheduler
from src.scheduling.sequence import Sequence, SequenceStatus
from src.sampling import SamplingParams


def _sched(
    *,
    num_blocks: int = 64,
    block_size: int = 16,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 256,
    eos: int = 42,
    enable_radix: bool = True,
) -> Scheduler:
    return Scheduler(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        num_blocks=num_blocks,
        block_size=block_size,
        eos_token_id=eos,
        enable_radix=enable_radix,
    )


def _seq(tokens: list[int], *, block_size: int = 16, **sp_kw) -> Sequence:
    kw = {"max_tokens": 8, **sp_kw}
    return Sequence(tokens, SamplingParams(**kw), block_size=block_size)


class TestScheduler:
    def test_prefill_then_decode(self):
        sch = _sched()
        seq = _seq([1, 2, 3, 4, 5])
        sch.add(seq)
        out = sch.schedule()
        assert out.is_prefill and len(out.seqs) == 1
        sch.postprocess(out.seqs, [99], True)
        assert seq.status == SequenceStatus.RUNNING
        assert seq.num_completion_tokens == 1
        out2 = sch.schedule()
        assert not out2.is_prefill and len(out2.seqs) == 1

    def test_chunked_prefill(self):
        sch = _sched(max_num_batched_tokens=4, block_size=16)
        seq = _seq([1] * 20)
        sch.add(seq)
        out1 = sch.schedule()
        assert out1.is_prefill and seq.num_scheduled_tokens == 4
        sch.postprocess(out1.seqs, [0], True)
        assert seq.num_completion_tokens == 0
        assert seq.num_cached_tokens == 4
        out2 = sch.schedule()
        assert out2.is_prefill

    def test_eos_finish(self):
        sch = _sched(eos=42)
        seq = _seq([1, 2, 3])
        sch.add(seq)
        out = sch.schedule()
        sch.postprocess(out.seqs, [42], True)
        assert seq.is_finished
        assert sch.is_finished()

    def test_max_tokens_zero_no_emit(self):
        sch = _sched()
        seq = _seq([1, 2, 3], max_tokens=0)
        sch.add(seq)
        out = sch.schedule()
        finished = sch.postprocess(out.seqs, [99], True)
        assert finished and seq.is_finished
        assert seq.num_completion_tokens == 0
        assert seq.token_ids == [1, 2, 3]

    def test_preempt_requeues(self):
        sch = _sched(num_blocks=16, block_size=16)
        s1 = _seq([1] * 32)
        s2 = _seq([2] * 32)
        sch.add(s1)
        sch.add(s2)
        out1 = sch.schedule()
        sch.postprocess(out1.seqs, [10], True)
        out2 = sch.schedule()
        sch.postprocess(out2.seqs, [11], True)
        assert s1.status == SequenceStatus.RUNNING
        assert s2.status == SequenceStatus.RUNNING
        sch.preempt(s2)
        assert s2.status == SequenceStatus.WAITING
        assert not s2.block_table
        assert s2 in sch.waiting

    def test_preempt_clears_mrope_state(self):
        from src.scheduling.sequence import MultiModalFeature
        import torch

        sch = _sched(num_blocks=16, block_size=16)
        mf = MultiModalFeature(
            modality="image",
            data=torch.zeros(4, 8),
            start=2,
            end=6,
            num_encoder_tokens=4,
            uid="x",
            grid_thw=torch.tensor([[1, 4, 4]]),
            mrope_origin=7,
        )
        seq = Sequence([1] * 16, SamplingParams(max_tokens=4), mm_features=[mf])
        seq.mrope_next = 99
        seq.rope_delta = -12
        sch.add(seq)
        out = sch.schedule()
        sch.postprocess(out.seqs, [0], True)
        assert seq.mrope_next == 99
        sch.preempt(seq)
        assert seq.mrope_next is None
        assert seq.rope_delta == 0
        assert seq.mm_features[0].mrope_origin is None
        assert seq.num_cached_tokens == 0

    def test_full_prefix_promote(self):
        sch = _sched(block_size=8, num_blocks=16)
        seq = _seq([7] * 12, block_size=8)
        sch.add(seq)
        n = sch.block_manager.can_allocate(seq)
        sch.block_manager.allocate(seq, n)
        seq.num_cached_tokens = seq.num_tokens
        out = sch.schedule()
        assert seq.status == SequenceStatus.RUNNING
        assert seq not in sch.waiting
        if out:
            assert not out.is_prefill or seq.num_scheduled_tokens >= 0
            if not out.is_prefill:
                assert seq in out.seqs
        else:
            out2 = sch.schedule()
            assert not out2.is_prefill and seq in out2.seqs

    def test_is_finished(self):
        sch = _sched()
        assert sch.is_finished()
        sch.add(_seq([1, 2]))
        assert not sch.is_finished()

    def test_abort_removes_and_frees(self):
        sch = _sched()
        seq = _seq([1, 2, 3, 4])
        sch.add(seq)
        out = sch.schedule()
        sch.postprocess(out.seqs, [7], True)
        assert seq.status == SequenceStatus.RUNNING
        assert seq.block_table
        sch.abort(seq)
        assert seq.is_finished
        assert sch.is_finished()
        assert not seq.block_table

    def test_sampling_params_not_flattened(self):
        seq = _seq([1, 2, 3], max_tokens=5, temperature=0.7)
        assert seq.sampling_params.max_tokens == 5
        assert seq.max_tokens == 5
        assert seq.temperature == 0.7
