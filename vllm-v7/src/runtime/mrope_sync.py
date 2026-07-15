from __future__ import annotations
from dataclasses import dataclass
from ..scheduling import Sequence

@dataclass
class MRopeStateUpdate:
    seq_id: int
    rope_delta: int
    mrope_next: int | None
    mrope_origins: tuple[int | None, ...]

def snapshot_mrope_updates(seqs: list[Sequence]) -> list[MRopeStateUpdate]:
    out: list[MRopeStateUpdate] = []
    for s in seqs:
        if not s.mm_features:
            continue
        out.append(MRopeStateUpdate(seq_id=s.seq_id, rope_delta=int(s.rope_delta), mrope_next=s.mrope_next, mrope_origins=tuple((mf.mrope_origin for mf in s.mm_features))))
    return out

def apply_mrope_updates(seqs: list[Sequence], updates: list[MRopeStateUpdate] | None) -> None:
    if not updates:
        return
    by_id = {s.seq_id: s for s in seqs}
    for u in updates:
        s = by_id.get(u.seq_id)
        if s is None:
            continue
        s.rope_delta = u.rope_delta
        s.mrope_next = u.mrope_next
        if len(u.mrope_origins) != len(s.mm_features):
            continue
        for mf, origin in zip(s.mm_features, u.mrope_origins):
            mf.mrope_origin = origin
