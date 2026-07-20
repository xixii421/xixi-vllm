from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ...scheduling import MultiModalFeature, Sequence
from ..input.features import overlaps_window


@dataclass(slots=True)
class DecodeSequenceMetadata:
    """Minimal Sequence view needed by workers for one decode step.

    Prompt/completion history and scheduler-only state stay in the engine
    process.  This keeps the steady-state TP/PP RPC payload proportional to
    the page table rather than the full token history.
    """

    seq_id: int
    num_tokens: int
    num_cached_tokens: int
    num_scheduled_tokens: int
    last_token: int
    block_size: int
    block_table: list[int]
    rope_delta: int
    temperature: float
    top_p: float | None
    top_k: int | None
    min_p: float | None
    mm_features: list[MultiModalFeature] = field(default_factory=list)

    @classmethod
    def from_sequence(cls, seq: Sequence) -> "DecodeSequenceMetadata":
        return cls(
            seq_id=seq.seq_id,
            num_tokens=seq.num_tokens,
            num_cached_tokens=seq.num_cached_tokens,
            num_scheduled_tokens=seq.num_scheduled_tokens,
            last_token=seq.last_token,
            block_size=seq.block_size,
            block_table=seq.block_table,
            rope_delta=seq.rope_delta,
            temperature=seq.temperature,
            top_p=seq.top_p,
            top_k=seq.top_k,
            min_p=seq.min_p,
        )

    def __len__(self) -> int:
        return self.num_tokens

    @property
    def last_block_num_tokens(self) -> int:
        return self.num_tokens - (len(self.block_table) - 1) * self.block_size

    @property
    def has_multimodal(self) -> bool:
        return bool(self.mm_features)


@dataclass(slots=True)
class PrefillSequenceWire:
    """Prefill RPC view: tokens + MM span meta, never pixel tensors.

    Pixels travel once via ``register_mm_pixels`` keyed by ``uid``.
    """

    seq_id: int
    token_ids: list[int]
    num_tokens: int
    num_cached_tokens: int
    num_scheduled_tokens: int
    last_token: int
    block_size: int
    block_table: list[int]
    rope_delta: int
    mrope_next: int | None
    temperature: float
    top_p: float | None
    top_k: int | None
    min_p: float | None
    mm_features: list[MultiModalFeature]

    @classmethod
    def from_sequence(cls, seq: Sequence) -> "PrefillSequenceWire":
        return cls(
            seq_id=seq.seq_id,
            token_ids=seq.token_ids,
            num_tokens=seq.num_tokens,
            num_cached_tokens=seq.num_cached_tokens,
            num_scheduled_tokens=seq.num_scheduled_tokens,
            last_token=seq.last_token,
            block_size=seq.block_size,
            block_table=list(seq.block_table),
            rope_delta=seq.rope_delta,
            mrope_next=seq.mrope_next,
            temperature=seq.temperature,
            top_p=seq.top_p,
            top_k=seq.top_k,
            min_p=seq.min_p,
            mm_features=[mf.meta_copy() for mf in seq.mm_features],
        )

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def has_multimodal(self) -> bool:
        return bool(self.mm_features)

    @property
    def last_block_num_tokens(self) -> int:
        if not self.block_table:
            return 0
        return self.num_tokens - (len(self.block_table) - 1) * self.block_size


def collect_mm_pixels(seqs: list[Sequence]) -> dict[str, torch.Tensor]:
    """Pixels for MM spans overlapping the current scheduled windows (once)."""
    out: dict[str, torch.Tensor] = {}
    for seq in seqs:
        cs = seq.num_cached_tokens
        ce = cs + seq.num_scheduled_tokens
        for mf in seq.mm_features:
            if mf.data is None or mf.uid in out:
                continue
            if overlaps_window(mf, cs, ce):
                out[mf.uid] = mf.data
    return out


def compact_execute_sequences(
    seqs: list[Sequence], is_prefill: bool
) -> list[PrefillSequenceWire] | list[DecodeSequenceMetadata]:
    """Strip engine-only state; never put pixel tensors on the execute wire."""
    if is_prefill:
        return [PrefillSequenceWire.from_sequence(seq) for seq in seqs]
    return [DecodeSequenceMetadata.from_sequence(seq) for seq in seqs]
