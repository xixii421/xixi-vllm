"""Unit tests: MRoPE Engine↔Worker sync helpers (no GPU)."""

from __future__ import annotations

from src.runtime.runner.mrope_sync import (
    MRopeStateUpdate,
    apply_mrope_updates,
    snapshot_mrope_updates,
)
from src.scheduling import MultiModalFeature, Sequence
from src.sampling import SamplingParams
import torch


def _seq_with_mm() -> Sequence:
    mf = MultiModalFeature(
        modality="image",
        data=torch.zeros(4, 3),
        start=1,
        end=5,
        num_encoder_tokens=4,
        uid="x",
        grid_thw=torch.tensor([[1, 2, 2]]),
        mrope_origin=None,
    )
    return Sequence([1, 2, 3, 4, 5], SamplingParams(max_tokens=4), mm_features=[mf])


def test_snapshot_and_apply_roundtrip():
    worker = _seq_with_mm()
    worker.rope_delta = 7
    worker.mrope_next = 42
    worker.mm_features[0].mrope_origin = 3

    updates = snapshot_mrope_updates([worker])
    assert len(updates) == 1
    assert updates[0].rope_delta == 7
    assert updates[0].mrope_next == 42
    assert updates[0].mrope_origins == (3,)

    engine = _seq_with_mm()
    engine.seq_id = worker.seq_id
    assert engine.rope_delta == 0
    apply_mrope_updates([engine], updates)
    assert engine.rope_delta == 7
    assert engine.mrope_next == 42
    assert engine.mm_features[0].mrope_origin == 3


def test_apply_ignores_unknown_seq_and_empty():
    s = _seq_with_mm()
    apply_mrope_updates([s], None)
    apply_mrope_updates(
        [s],
        [MRopeStateUpdate(seq_id=s.seq_id + 99, rope_delta=1, mrope_next=2, mrope_origins=(0,))],
    )
    assert s.rope_delta == 0


def test_text_only_seq_not_snapshotted():
    s = Sequence([1, 2, 3], SamplingParams(max_tokens=2))
    s.rope_delta = 9
    assert snapshot_mrope_updates([s]) == []
