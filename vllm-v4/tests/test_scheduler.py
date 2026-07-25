"""Scheduler: prefill / decode / chunk / eos / preempt / mix / prefix."""

from __future__ import annotations

import torch

from src.engine.scheduler import Scheduler
from src.memory import KVManager, Sequence, SequenceStatus
from src.sampling_params import SamplingParams


def _fake_hf():
    class C:
        num_key_value_heads = 2
        num_attention_heads = 4
        hidden_size = 64
        head_dim = 16
        num_hidden_layers = 2

    return C()


def _sched(
    *,
    num_blocks: int = 64,
    block_size: int = 16,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 256,
    eos: int = 42,
    mix_prefill_decode: bool = False,
    prefix_backend: str = "hash",
) -> Scheduler:
    kv = KVManager(
        _fake_hf(),
        num_blocks=num_blocks,
        block_size=block_size,
        device=torch.device("cpu"),
        dtype=torch.float16,
        prefix_backend=prefix_backend,  # type: ignore[arg-type]
        reserve_dummy_page=False,
    )
    return Scheduler(
        kv,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos_token_id=eos,
        mix_prefill_decode=mix_prefill_decode,
    )


def _seq(tokens: list[int], *, block_size: int = 16, **sp_kw) -> Sequence:
    kw = {"max_tokens": 8, **sp_kw}
    return Sequence.from_prompt(
        tokens, block_size=block_size, sampling=SamplingParams(**kw)
    )


class TestScheduler:
    def test_prefill_then_decode(self):
        sch = _sched()
        seq = _seq([1, 2, 3, 4, 5])
        sch.add(seq)
        prefills, decodes = sch.schedule()
        assert prefills and not decodes
        sch.postprocess(prefills, [99], is_prefill=True)
        assert seq.status == SequenceStatus.RUNNING
        prefills2, decodes2 = sch.schedule()
        assert not prefills2 and len(decodes2) == 1

    def test_chunked_prefill(self):
        sch = _sched(max_num_batched_tokens=4, block_size=16)
        seq = _seq([1] * 20)
        sch.add(seq)
        p1, d1 = sch.schedule()
        assert p1 and not d1 and seq.num_scheduled_tokens == 4
        sch.postprocess(p1, [0], is_prefill=True)
        assert seq.num_completion_tokens == 0
        assert seq.num_cached_tokens == 4
        p2, d2 = sch.schedule()
        assert p2 and not d2

    def test_eos_finish(self):
        sch = _sched(eos=42)
        seq = _seq([1, 2, 3])
        sch.add(seq)
        p, _ = sch.schedule()
        sch.postprocess(p, [42], is_prefill=True)
        assert seq.is_finished

    def test_max_tokens_zero_no_emit(self):
        sch = _sched()
        seq = _seq([1, 2, 3], max_tokens=0)
        sch.add(seq)
        p, _ = sch.schedule()
        finished = sch.postprocess(p, [99], is_prefill=True)
        assert finished and seq.is_finished
        assert seq.num_completion_tokens == 0

    def test_preempt_requeues(self):
        sch = _sched(num_blocks=16, block_size=16)
        s1 = _seq([1] * 32)
        s2 = _seq([2] * 32)
        sch.add(s1)
        sch.add(s2)
        p1, _ = sch.schedule()
        sch.postprocess(p1, [10], is_prefill=True)
        p2, _ = sch.schedule()
        sch.postprocess(p2, [11], is_prefill=True)
        sch.preempt(s2)
        assert s2.status == SequenceStatus.WAITING
        assert not s2.block_table

    def test_hash_full_page_second_hit(self):
        sch = _sched(block_size=8, num_blocks=32, prefix_backend="hash")
        tokens = [7] * 16
        s1 = _seq(tokens, block_size=8)
        sch.add(s1)
        p, _ = sch.schedule()
        sch.postprocess(p, [1], is_prefill=True)
        while not s1.is_finished:
            _, d = sch.schedule()
            if not d:
                break
            sch.postprocess(d, [42], is_prefill=False)

        s2 = _seq(tokens, block_size=8, max_tokens=1)
        assert sch.kv.can_allocate(s2)
        sch.kv.allocate(s2)
        assert s2.num_cached_tokens == 16

    def test_radix_second_hit(self):
        sch = _sched(block_size=8, num_blocks=32, prefix_backend="radix")
        tokens = [3] * 16
        s1 = _seq(tokens, block_size=8)
        sch.add(s1)
        p, _ = sch.schedule()
        sch.postprocess(p, [1], is_prefill=True)
        while not s1.is_finished:
            _, d = sch.schedule()
            if not d:
                break
            sch.postprocess(d, [42], is_prefill=False)

        s2 = _seq(tokens, block_size=8, max_tokens=2)
        sch.add(s2)
        # Full page hit → promote then decode in same schedule().
        p, d = sch.schedule()
        assert not p and d and d[0] is s2
        assert s2.num_cached_tokens == 16
        assert s2.status == SequenceStatus.RUNNING

    def test_hash_full_hit_promotes_to_decode(self):
        sch = _sched(block_size=8, num_blocks=32, prefix_backend="hash")
        tokens = [9] * 16
        s1 = _seq(tokens, block_size=8)
        sch.add(s1)
        p, _ = sch.schedule()
        sch.postprocess(p, [1], is_prefill=True)
        while not s1.is_finished:
            _, d = sch.schedule()
            if not d:
                break
            sch.postprocess(d, [7], is_prefill=False)

        s2 = _seq(tokens, block_size=8, max_tokens=1)
        sch.add(s2)
        p, d = sch.schedule()
        assert not p and d and d[0] is s2
        assert s2.num_cached_tokens == 16

    def test_mix_decode_with_prefill_chunk(self):
        sch = _sched(max_num_batched_tokens=8, mix_prefill_decode=True)
        s_run = _seq([1, 2, 3, 4])
        s_wait = _seq([9] * 16)
        sch.add(s_run)
        p, _ = sch.schedule()
        sch.postprocess(p, [10], is_prefill=True)
        sch.add(s_wait)
        prefills, decodes = sch.schedule()
        assert s_run in decodes and s_wait in prefills
        assert s_wait.num_scheduled_tokens == 7
