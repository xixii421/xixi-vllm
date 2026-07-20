"""Scheduler → Engine boundary (ullm-style compressed step result)."""

from __future__ import annotations

from dataclasses import dataclass

from .sequence import Sequence


@dataclass(slots=True, frozen=True)
class SchedulerOutput:
    """One scheduling decision.

    ``seqs`` are live engine-owned objects (not RPC wire format). Decode RPC
    still goes through ``runtime.execute.request.DecodeSequenceMetadata``.
    """

    seqs: tuple[Sequence, ...]
    is_prefill: bool

    @classmethod
    def empty(cls) -> SchedulerOutput:
        return cls(seqs=(), is_prefill=False)

    def __bool__(self) -> bool:
        return bool(self.seqs)

    @property
    def metric(self) -> int:
        if not self.seqs:
            return 0
        if self.is_prefill:
            return sum(s.num_scheduled_tokens for s in self.seqs)
        return -len(self.seqs)
