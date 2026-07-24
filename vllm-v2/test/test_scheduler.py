"""CPU tests for control-plane Scheduler."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.engine.scheduler import Scheduler
from src.kv.manager import KVManager
from src.kv.sequence import Sequence
from src.sampling_params import SamplingParams


class _FakeAttn(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_kv_heads = 1
        self.head_dim = 4


class _FakeLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeAttn()


class _FakeInner(torch.nn.Module):
    def __init__(self, n: int = 2) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeLayer() for _ in range(n)])


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeInner()


def _sched(**kw) -> Scheduler:
    model = _FakeModel()
    kv = KVManager(
        model,
        num_blocks=64,
        block_size=4,
        prefix_backend="none",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    defaults = dict(
        max_num_seqs=4,
        max_num_batched_tokens=128,
        eos_token_id=2,
        mix_prefill_decode=False,
    )
    defaults.update(kw)
    return Scheduler(kv, **defaults)


def test_prefill_then_decode() -> None:
    sch = _sched()
    seq = Sequence.from_prompt(
        [1, 2, 3, 4, 5], block_size=4, sampling=SamplingParams(max_tokens=3)
    )
    sch.add(seq)
    prefills, decodes = sch.schedule()
    assert prefills == [seq] and not decodes
    assert seq.status.name == "RUNNING"
    sch.postprocess(prefills, [9], is_prefill=True)
    assert seq.token_ids[-1] == 9
    prefills2, decodes2 = sch.schedule()
    assert not prefills2 and decodes2 == [seq]
    print("  [PASS] prefill then decode")


def test_multi_waiting_prefills() -> None:
    sch = _sched(max_num_seqs=2)
    a = Sequence.from_prompt([1, 2, 3], block_size=4, sampling=SamplingParams(max_tokens=1))
    b = Sequence.from_prompt([4, 5, 6], block_size=4, sampling=SamplingParams(max_tokens=1))
    sch.add(a)
    sch.add(b)
    prefills, decodes = sch.schedule()
    assert len(prefills) == 2 and not decodes
    print("  [PASS] multi waiting prefills")


def test_chunked_prefill() -> None:
    sch = _sched(max_num_batched_tokens=3)
    seq = Sequence.from_prompt(
        list(range(10)), block_size=4, sampling=SamplingParams(max_tokens=1)
    )
    sch.add(seq)
    p1, d1 = sch.schedule()
    assert p1 == [seq] and not d1 and seq.num_scheduled_tokens == 3
    assert seq.status.name == "WAITING"
    sch.postprocess(p1, [0], is_prefill=True)
    assert seq.num_cached_tokens == 3 and len(seq.token_ids) == 10
    p2, _ = sch.schedule()
    assert seq.num_scheduled_tokens == 3
    print("  [PASS] chunked prefill")


def test_mix_decode_and_prefill() -> None:
    sch = _sched(max_num_batched_tokens=8, mix_prefill_decode=True)
    run = Sequence.from_prompt(
        [1, 2, 3, 4], block_size=4, sampling=SamplingParams(max_tokens=8)
    )
    sch.add(run)
    p0, d0 = sch.schedule()
    assert not d0 and p0 == [run] and run.status.name == "RUNNING"
    sch.postprocess(p0, [10], is_prefill=True)

    long = Sequence.from_prompt(
        list(range(20, 40)), block_size=4, sampling=SamplingParams(max_tokens=2)
    )
    sch.add(long)
    prefills, decodes = sch.schedule()
    assert decodes == [run] and run.num_scheduled_tokens == 1
    assert prefills == [long] and long.num_scheduled_tokens == 7  # budget 8-1
    print("  [PASS] mix decode + prefill chunk")


def test_finish_on_eos() -> None:
    sch = _sched(eos_token_id=99)
    seq = Sequence.from_prompt(
        [1, 2, 3, 4], block_size=4, sampling=SamplingParams(max_tokens=8)
    )
    sch.add(seq)
    prefills, _ = sch.schedule()
    finished = sch.postprocess(prefills, [99], is_prefill=True)
    assert finished and seq.is_finished and sch.is_finished()
    print("  [PASS] finish on eos")


def main() -> None:
    print("── Scheduler ──")
    test_prefill_then_decode()
    test_multi_waiting_prefills()
    test_chunked_prefill()
    test_mix_decode_and_prefill()
    test_finish_on_eos()
    print("ALL_OK")


if __name__ == "__main__":
    main()
