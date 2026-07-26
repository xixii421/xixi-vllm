"""Request state owned by the scheduler (and shared with the engine loop).

Keep this object lean:
  - tokens + KV page table + schedule counters  → scheduler / runner
  - ``sampling_params``                          → sampler (via properties)
  - ``mm_features``                              → runtime input/vision (CPU meta;
    pixels are registered once via RPC, not embedded in every execute payload)
  - mRoPE scalars                                → vision / mrope_sync
"""

from __future__ import annotations

from copy import copy
from enum import Enum, auto
from itertools import count

import torch

from ..sampling import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class MultiModalFeature:
    """CPU modality span. Prefer ``token_end`` over a possibly-stale ``end``."""

    __slots__ = (
        "modality",
        "data",
        "start",
        "end",
        "num_encoder_tokens",
        "uid",
        "grid_thw",
        "grid_thw_tuple",
        "mrope_origin",
    )

    def __init__(
        self,
        modality: str,
        data: torch.Tensor | None,
        start: int,
        end: int,
        num_encoder_tokens: int,
        uid: str,
        grid_thw: torch.Tensor | None = None,
        grid_thw_tuple: tuple[int, int, int] | None = None,
        mrope_origin: int | None = None,
    ) -> None:
        self.modality = modality
        self.data = data
        self.start = start
        self.end = end
        self.num_encoder_tokens = num_encoder_tokens
        self.uid = uid
        self.grid_thw = grid_thw
        if grid_thw_tuple is not None:
            self.grid_thw_tuple = grid_thw_tuple
        elif grid_thw is not None:
            flat = grid_thw[0].tolist() if grid_thw.ndim == 2 else grid_thw.tolist()
            self.grid_thw_tuple = (int(flat[0]), int(flat[1]), int(flat[2]))
        else:
            self.grid_thw_tuple = None
        self.mrope_origin = mrope_origin

    @property
    def token_end(self) -> int:
        return self.start + self.num_encoder_tokens

    def meta_copy(self) -> "MultiModalFeature":
        """Span metadata without pixel tensors (safe for RPC wire)."""
        return MultiModalFeature(
            modality=self.modality,
            data=None,
            start=self.start,
            end=self.end,
            num_encoder_tokens=self.num_encoder_tokens,
            uid=self.uid,
            grid_thw=self.grid_thw,
            grid_thw_tuple=self.grid_thw_tuple,
            mrope_origin=self.mrope_origin,
        )


def slot_of(block_table: list[int], pos: int, block_size: int) -> int:
    block_idx = pos // block_size
    if block_idx >= len(block_table):
        raise IndexError(
            f"pos={pos} needs block_idx={block_idx}, but block_table has "
            f"{len(block_table)} pages"
        )
    return block_table[block_idx] * block_size + pos % block_size


def slots_for_range(
    block_table: list[int], start: int, end: int, block_size: int
) -> list[int]:
    return [slot_of(block_table, pos, block_size) for pos in range(start, end)]


class Sequence:
    _counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        mm_features: list[MultiModalFeature] | None = None,
        *,
        block_size: int = 256,
    ) -> None:
        if not token_ids:
            raise ValueError("empty token_ids")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.seq_id = next(Sequence._counter)
        self.status = SequenceStatus.WAITING
        self.block_size = int(block_size)
        self.sampling_params = sampling_params or SamplingParams()

        self.token_ids: list[int] = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table: list[int] = []

        self.mm_features: list[MultiModalFeature] = list(mm_features or [])
        self.rope_delta: int = 0
        self.mrope_next: int | None = None

    # --- sampling (compat properties; prefer ``sampling_params``) ---
    @property
    def temperature(self) -> float:
        return self.sampling_params.temperature

    @property
    def top_p(self) -> float | None:
        return self.sampling_params.top_p

    @property
    def top_k(self) -> int | None:
        return self.sampling_params.top_k

    @property
    def min_p(self) -> float | None:
        return self.sampling_params.min_p

    @property
    def max_tokens(self) -> int:
        return self.sampling_params.max_tokens

    @property
    def ignore_eos(self) -> bool:
        return self.sampling_params.ignore_eos

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
    def first_mm_token(self) -> int | None:
        if not self.mm_features:
            return None
        return min(int(mf.start) for mf in self.mm_features)

    def prefix_cacheable_full_blocks(self) -> int:
        """Full pages entirely before the first image token (text-only prefix).

        Pages that overlap or follow multimodal spans are not published/matched
        with token-ids alone (image content is not in the token stream).
        """
        n_full = max(self.num_blocks - 1, 0)
        first = self.first_mm_token
        if first is None:
            return n_full
        return min(n_full, int(first) // self.block_size)

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
        self.rope_delta = 0
        self.mrope_next = None
        for mf in self.mm_features:
            mf.mrope_origin = None
