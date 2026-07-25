from __future__ import annotations

import pytest
import torch

from src.scheduling import Sequence
from src.sampling import SamplingParams


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
    return Sequence(
        list(tokens),
        sp or SamplingParams(max_tokens=8),
        block_size=block_size,
    )


needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)
needs_2gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs >=2 GPUs",
)
needs_4gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="needs >=4 GPUs",
)


def pytest_collection_modifyitems(config, items) -> None:
    for item in items:
        path = str(item.fspath).replace("\\", "/")
        if "/tests/unit/" in path:
            item.add_marker(pytest.mark.unit)
        elif "/tests/gpu/" in path:
            item.add_marker(pytest.mark.gpu)
        elif "/tests/distributed/" in path:
            item.add_marker(pytest.mark.distributed)
