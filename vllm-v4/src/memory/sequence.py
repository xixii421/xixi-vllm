from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count
from typing import Optional

from ..sampling_params import SamplingParams

import torch

_SEQ_IDS = count(1)


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class MultiModalFeature:
    modality: str  # "image" | "video"
    data: "torch.Tensor"  # pixel_values on CPU
    start: int
    end: int
    num_encoder_tokens: int
    uid: str
    grid_thw: Optional["torch.Tensor"] = None
    # Cached DeepStack features [ndps][num_encoder_tokens, H] (GPU/CPU ok).
    deepstack: Optional[list["torch.Tensor"]] = None
    # Absolute MRoPE origin of this image (set on the chunk that starts it).
    mrope_origin: Optional[int] = None


def slot_of(block_table: list[int], pos: int, block_size: int) -> int:
    block_idx = pos // block_size
    if block_idx >= len(block_table):
        raise IndexError(
            f"pos={pos} needs block_idx={block_idx}, but block_table has {len(block_table)} pages"
        )
    return block_table[block_idx] * block_size + (pos % block_size)


def slots_for_range(
    block_table: list[int],
    start: int,
    end: int,
    block_size: int,
) -> list[int]:
    return [slot_of(block_table, pos, block_size) for pos in range(start, end)]


@dataclass
class Sequence:
    """One request: tokens + block_table + scheduling cursors (v2-aligned)."""

    token_ids: list[int]
    block_size: int
    seq_id: int = 0
    status: SequenceStatus = SequenceStatus.WAITING
    block_table: list[int] = field(default_factory=list)
    num_cached_tokens: int = 0
    num_scheduled_tokens: int = 0
    is_prefill: bool = True
    num_prompt_tokens: int = 0
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int = 128
    ignore_eos: bool = False
    last_token: int = 0
    # --- Multimodal (v4) ---
    mm_features: list[MultiModalFeature] = field(default_factory=list)
    # Qwen3-VL decode position = (len-1) + rope_delta (set after mm prefill).
    rope_delta: int = 0
    # Absolute next MRoPE index on all 3 axes (chunked-prefill continuation).
    mrope_next: int | None = None

    @classmethod
    def from_prompt(
        cls,
        token_ids: list[int],
        *,
        block_size: int,
        sampling: SamplingParams | None = None,
        mm_features: list[MultiModalFeature] | None = None,
    ) -> Sequence:
        sp = sampling or SamplingParams()
        ids = copy(token_ids)
        if not ids:
            raise ValueError("empty prompt")
        return cls(
            token_ids=ids,
            block_size=block_size,
            seq_id=next(_SEQ_IDS),
            num_prompt_tokens=len(ids),
            last_token=ids[-1],
            temperature=sp.temperature,
            top_p=sp.top_p,
            top_k=sp.top_k,
            min_p=sp.min_p,
            max_tokens=sp.max_tokens,
            ignore_eos=sp.ignore_eos,
            mm_features=list(mm_features or []),
        )

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def num_blocks(self) -> int:
        n = self.num_tokens
        if n == 0:
            return 0
        return (n + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        if self.num_blocks == 0:
            return 0
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def num_full_blocks(self) -> int:
        return self.num_tokens // self.block_size

    @property
    def has_multimodal(self) -> bool:
        return bool(self.mm_features)

    @property
    def first_mm_token(self) -> int:
        """Token offset of the earliest image placeholder (num_tokens if none)."""
        if not self.mm_features:
            return self.num_tokens
        return min(mf.start for mf in self.mm_features)

    def prefix_cacheable_full_blocks(self) -> int:
        """Full blocks eligible for APC.

        For text this is every full block. For multimodal we only cache blocks
        that lie *entirely before* the first image placeholder: image KV depends
        on pixel content that the token-id block hash does not capture, so caching
        such blocks by token ids alone would return wrong KV on a hash collision.
        """
        n_full = self.num_full_blocks()
        if not self.mm_features:
            return n_full
        return min(n_full, self.first_mm_token // self.block_size)

    def reset_mrope_state(self) -> None:
        """Clear MRoPE cursors so a full re-prefill after deallocate is correct.

        Preempt / finished deallocate resets ``num_cached_tokens`` but must also
        drop ``mrope_next`` / ``rope_delta`` / per-image ``mrope_origin``; otherwise
        chunked position math continues from a stale absolute cursor.
        """
        self.rope_delta = 0
        self.mrope_next = None
        for mf in self.mm_features:
            mf.mrope_origin = None

    def block_token_ids(self, i: int) -> list[int]:
        assert 0 <= i < self.num_blocks
        start = i * self.block_size
        end = min(start + self.block_size, self.num_tokens)
        return self.token_ids[start:end]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.last_token = token_id
