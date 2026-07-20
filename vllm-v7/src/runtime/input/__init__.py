"""Stage 1 — request preprocess (engine process, CPU).

``add_request`` path: HF processor → typed spans → ``MultiModalFeature`` list.
Does not touch GPU, ViT, or EncoderCache.
"""

from .features import (
    ImageSpanInfo,
    build_mm_features,
    overlaps_window,
    span_end,
    truncate_mm_features,
)
from .hasher import hash_image_pixels
from .processor import MultiModalProcessor

__all__ = [
    "ImageSpanInfo",
    "MultiModalProcessor",
    "build_mm_features",
    "hash_image_pixels",
    "overlaps_window",
    "span_end",
    "truncate_mm_features",
]
