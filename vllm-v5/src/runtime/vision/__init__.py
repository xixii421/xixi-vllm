"""Stage 3b — worker vision encode / merge (GPU, prefill only).

Called by ``runner.ModelRunner`` via injected ``model.mm_encoder``.
Depends on ``layers.vision`` for ViT bricks; uses ``input.features`` for span helpers.
"""

from .encoder_cache import EncoderCache
from .encoder_runner import VisionEncoderRunner
from .pixel_store import PixelStore
from .positions import (
    MRopeSeqUpdate,
    MRopeStateUpdate,
    apply_mrope_seq_updates,
    apply_mrope_updates,
    compute_mrope_positions,
    mrope_positions,
)

__all__ = [
    "EncoderCache",
    "PixelStore",
    "VisionEncoderRunner",
    "MRopeSeqUpdate",
    "MRopeStateUpdate",
    "apply_mrope_seq_updates",
    "apply_mrope_updates",
    "compute_mrope_positions",
    "mrope_positions",
]
