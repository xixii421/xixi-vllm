from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from enum import Enum, auto
from itertools import count

import torch

from ..sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class MultiModalFeature:
    modality: str  # "image" | "video"
    data: torch.Tensor  # pixel_values on CPU
    start: int
    end: int
    num_encoder_tokens: int
    uid: str
    grid_thw: torch.Tensor | None = None
    # Cached DeepStack features [ndps][num_encoder_tokens, H] (GPU/CPU ok).
    deepstack: list[torch.Tensor] | None = None
    mrope_origin: int | None = None


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


class Sequence:

    block_size = 256
    _counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        mm_features: list[MultiModalFeature] | None = None,
    ) -> None:
        if not token_ids:
            raise ValueError("empty token_ids")
        sp = sampling_params or SamplingParams()
        self.seq_id = next(Sequence._counter)
        self.status = SequenceStatus.WAITING
        self.token_ids: list[int] = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table: list[int] = []
        self.mm_features: list[MultiModalFeature] = list(mm_features or [])
        # Qwen3-VL: decode position = (len-1) + rope_delta (set after mm prefill).
        self.rope_delta: int = 0
        # Absolute next MRoPE index on all 3 axes (for chunked prefill continuation).
        self.mrope_next: int | None = None

        self.temperature = sp.temperature
        self.top_p = sp.top_p
        self.top_k = sp.top_k
        self.min_p = sp.min_p
        self.max_tokens = sp.max_tokens
        self.ignore_eos = sp.ignore_eos

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def has_multimodal(self) -> bool:
        return bool(self.mm_features)

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def num_blocks(self) -> int:
        if self.num_tokens == 0:
            return 0
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        if self.num_blocks == 0:
            return 0
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block_token_ids(self, i: int) -> list[int]:
        assert 0 <= i < self.num_blocks
        start = i * self.block_size
        end = min(start + self.block_size, self.num_tokens)
        return self.token_ids[start:end]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def reset_mrope_state(self) -> None:
        """Clear MRoPE cursor so a full re-prefill after KV deallocate is correct.

        Preempt / finished deallocate resets ``num_cached_tokens`` but must also
        drop ``mrope_next`` / ``rope_delta`` / per-image ``mrope_origin``;
        otherwise chunked position math continues from a stale absolute cursor.
        """
        self.rope_delta = 0
        self.mrope_next = None
        for mf in self.mm_features:
            mf.mrope_origin = None
