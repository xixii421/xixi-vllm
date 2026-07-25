"""Load vllm-v7 MoE CUDA aux ops (align / silu / moe_sum).

Official vLLM ships ``_moe_C`` against torch>=2.11 stable ABI; v7 uses torch 2.9,
so we compile a classic-ATen port under ``csrc/moe/``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

_CSRC = Path(__file__).resolve().parents[3] / "csrc" / "moe"


@lru_cache(maxsize=1)
def moe_cuda_available() -> bool:
    if os.environ.get("VLLM_V7_MOE_CUDA", "1") in ("0", "false", "False"):
        return False
    try:
        _load()
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _load():
    from torch.utils.cpp_extension import load

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        for cand in ("/usr/local/cuda-12.8", "/usr/local/cuda"):
            if Path(cand).exists():
                os.environ["CUDA_HOME"] = cand
                break

    src = str(_CSRC / "moe_kernels.cu")
    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]
    # Ampere / Ada always; Blackwell SM120 needs toolkit ≥12.8 (arch 120a).
    # Do not emit 120f here — that needs nvcc ≥12.9.
    extra_cuda_cflags += [
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_120a,code=sm_120a",
    ]
    return load(
        name="vllm_v7_moe_cuda",
        sources=[src],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=bool(os.environ.get("VLLM_V7_MOE_CUDA_VERBOSE")),
        with_cuda=True,
    )


def moe_align_block_size_cuda(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    _load().moe_align_block_size(
        topk_ids,
        int(num_experts),
        int(block_size),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
    )


def moe_sum_cuda(input: torch.Tensor, output: torch.Tensor) -> None:
    """``input [M,K,H] -> output [M,H]`` (in-place into ``output``)."""
    _load().moe_sum(input, output)


def silu_and_mul_cuda(out: torch.Tensor, input: torch.Tensor) -> None:
    """``[gate|up]`` layout."""
    _load().silu_and_mul(out, input)


def mul_and_silu_cuda(out: torch.Tensor, input: torch.Tensor) -> None:
    """``[up|gate]`` layout → ``up * silu(gate)``."""
    _load().mul_and_silu(out, input)
