"""Snapshot worker-side mRoPE state for Engine Sequence sync over RPC."""

from __future__ import annotations

from ...scheduling import Sequence
from ..vision.positions import MRopeStateUpdate, apply_mrope_updates

__all__ = [
    "MRopeStateUpdate",
    "apply_mrope_updates",
    "snapshot_mrope_updates",
]


def snapshot_mrope_updates(seqs: list[Sequence]) -> list[MRopeStateUpdate]:
    out: list[MRopeStateUpdate] = []
    for s in seqs:
        if not s.mm_features:
            continue
        out.append(
            MRopeStateUpdate(
                seq_id=s.seq_id,
                rope_delta=int(s.rope_delta),
                mrope_next=s.mrope_next,
                mrope_origins=tuple(mf.mrope_origin for mf in s.mm_features),
            )
        )
    return out
