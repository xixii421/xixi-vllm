"""CPU tests for control-plane Scheduler."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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
    defaults = dict(max_num_seqs=4, max_num_batched_tokens=128, eos_token_id=2)
    defaults.update(kw)
    return Scheduler(kv, **defaults)


def test_prefill_then_decode() -> None:
    sch = _sched()
    seq = Sequence.from_prompt(
        [1, 2, 3, 4, 5], block_size=4, sampling=SamplingParams(max_tokens=3)
    )
    sch.add(seq)
    batch, is_prefill = sch.schedule()
    assert is_prefill and batch == [seq]
    assert seq.status.name == "RUNNING"
    sch.postprocess(batch, [9], True)
    assert seq.token_ids[-1] == 9
    batch2, is_prefill2 = sch.schedule()
    assert not is_prefill2 and batch2 == [seq]
    print("  [PASS] prefill then decode")


def test_multi_waiting_prefills() -> None:
    sch = _sched(max_num_seqs=2)
    a = Sequence.from_prompt([1, 2, 3], block_size=4, sampling=SamplingParams(max_tokens=1))
    b = Sequence.from_prompt([4, 5, 6], block_size=4, sampling=SamplingParams(max_tokens=1))
    sch.add(a)
    sch.add(b)
    batch, is_prefill = sch.schedule()
    assert is_prefill and len(batch) == 2
    print("  [PASS] multi waiting prefills")


def test_finish_on_eos() -> None:
    sch = _sched(eos_token_id=99)
    seq = Sequence.from_prompt(
        [1, 2, 3, 4], block_size=4, sampling=SamplingParams(max_tokens=8)
    )
    sch.add(seq)
    batch, _ = sch.schedule()
    finished = sch.postprocess(batch, [99], True)
    assert finished and seq.is_finished and sch.is_finished()
    print("  [PASS] finish on eos")


def main() -> None:
    print("── Scheduler ──")
    test_prefill_then_decode()
    test_multi_waiting_prefills()
    test_finish_on_eos()
    print("ALL_OK")


if __name__ == "__main__":
    main()
