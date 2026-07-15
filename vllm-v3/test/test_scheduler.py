"""CPU tests for Scheduler waiting/running + prefill-first."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.scheduler import Scheduler
from src.memory import Sequence
from src.sampling_params import SamplingParams


def _sched(**kw) -> Scheduler:
    defaults = dict(
        max_num_seqs=4,
        max_num_batched_tokens=128,
        num_blocks=64,
        block_size=4,
        eos_token_id=2,
    )
    defaults.update(kw)
    return Scheduler(**defaults)


def test_prefill_then_decode() -> None:
    sch = _sched()
    seq = Sequence([1, 2, 3, 4, 5], SamplingParams(max_tokens=3, temperature=0.0))
    sch.add(seq)

    batch, is_prefill = sch.schedule()
    assert is_prefill and batch == [seq]
    assert seq.status.name == "RUNNING"
    # Pretend model returned token 9
    sch.postprocess(batch, [9], True)
    assert seq.token_ids[-1] == 9
    assert not seq.is_finished

    batch2, is_prefill2 = sch.schedule()
    assert not is_prefill2
    assert batch2 == [seq]
    assert seq.num_scheduled_tokens == 1
    sch.postprocess(batch2, [10], False)

    print("  [PASS] prefill then decode")


def test_multi_waiting_prefills() -> None:
    sch = _sched(max_num_seqs=2)
    a = Sequence([1, 2, 3], SamplingParams(max_tokens=1))
    b = Sequence([4, 5, 6], SamplingParams(max_tokens=1))
    sch.add(a)
    sch.add(b)
    batch, is_prefill = sch.schedule()
    assert is_prefill and len(batch) == 2
    print("  [PASS] multi waiting prefills")


def test_finish_on_eos() -> None:
    sch = _sched(eos_token_id=99)
    seq = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=8))
    sch.add(seq)
    batch, _ = sch.schedule()
    finished = sch.postprocess(batch, [99], True)
    assert finished and seq.is_finished
    assert sch.is_finished()
    print("  [PASS] finish on eos")


def main() -> None:
    print("── Scheduler ──")
    test_prefill_then_decode()
    test_multi_waiting_prefills()
    test_finish_on_eos()
    print("\n✅ Scheduler tests passed.")


if __name__ == "__main__":
    main()
