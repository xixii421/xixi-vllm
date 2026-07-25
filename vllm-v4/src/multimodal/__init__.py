"""v4 multimodal subsystem (Qwen3-VL).

CPU: `MultiModalProcessor` (HF AutoProcessor → ids + pixels + spans).
GPU: `Qwen3VisionTower` + `EncoderCache`; `mrope_positions` builds LM MRoPE.
"""

from ._mrope import mrope_positions
from .processor import MultiModalProcessor
from .vision import EncoderCache, Qwen3VisionTower

__all__ = [
    "MultiModalProcessor",
    "Qwen3VisionTower",
    "EncoderCache",
    "mrope_positions",
]
