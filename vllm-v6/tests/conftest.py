"""Pytest fixtures for vllm-v4."""

from __future__ import annotations

import pytest
import torch

from src.memory import Sequence
from src.sampling_params import SamplingParams


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    return torch.cuda.is_available()


@pytest.fixture
def block_size() -> int:
    return 16


def make_seq(
    tokens: list[int],
    *,
    block_size: int = 16,
    sp: SamplingParams | None = None,
) -> Sequence:
    Sequence.block_size = block_size
    return Sequence(list(tokens), sp or SamplingParams(max_tokens=8))
