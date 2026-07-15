"""Engine smoke helpers shared by GPU / distributed tests."""

from __future__ import annotations

import gc
import os
from typing import Any

import torch
from PIL import Image

from src import Engine, EngineConfig, SamplingParams

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def free_cuda() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def cfg(model: str | os.PathLike[str], **kwargs: Any) -> EngineConfig:
    """Build EngineConfig with sensible test defaults (kwargs override)."""
    defaults: dict[str, Any] = dict(
        model=str(model),
        enforce_eager=True,
        max_num_seqs=4,
        context_len=1024,
        num_kvcache_blocks=32,
        gpu_memory_utilization=0.75,
        nccl_port=0,
        enable_vit_cuda_graph=False,
    )
    defaults.update(kwargs)
    return EngineConfig(**defaults)


def solid_image(size: tuple[int, int] = (64, 64), color=(32, 128, 200)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def greedy(
    config: EngineConfig,
    prompt: str,
    *,
    n: int = 8,
    images=None,
) -> dict[str, Any]:
    """Run one greedy request; return tokens + multimodal RoPE state."""
    free_cuda()
    eng = Engine(config)
    try:
        seq = eng.add_request(
            prompt,
            SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True),
            images=images,
        )
        while not seq.is_finished:
            eng.step()
        return {
            "tokens": seq.token_ids[seq.num_prompt_tokens :],
            "rope_delta": int(seq.rope_delta),
            "mrope_next": seq.mrope_next,
            "origins": [mf.mrope_origin for mf in seq.mm_features],
        }
    finally:
        eng.destroy()
        free_cuda()


def greedy_tokens(
    config: EngineConfig,
    prompt: str,
    *,
    n: int = 8,
    images=None,
) -> list[int]:
    return greedy(config, prompt, n=n, images=images)["tokens"]
