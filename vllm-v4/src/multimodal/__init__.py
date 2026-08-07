"""v4 multimodal subsystem (Qwen3-VL).

CPU: `MultiModalProcessor` (per-image mm_kwargs cache + prompt assembly).
GPU: `Qwen3VisionTower` + `EncoderCache`; `mrope_positions` builds LM MRoPE.
"""

from __future__ import annotations

from ._mrope import mrope_positions
from .hasher import hash_image_pixels
from .processor import MMPlaceholder, MultiModalInputs, MultiModalProcessor
from .vision import EncoderCache, Qwen3VisionTower, VisionEncodeOutput

__all__ = [
    "MultiModalProcessor",
    "MultiModalInputs",
    "MMPlaceholder",
    "hash_image_pixels",
    "Qwen3VisionTower",
    "EncoderCache",
    "VisionEncodeOutput",
    "mrope_positions",
]
