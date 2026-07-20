"""Typed multimodal request helpers shared by Engine and EncoderRunner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import torch

from .hasher import hash_image_pixels

if TYPE_CHECKING:
    from ...scheduling.sequence import MultiModalFeature  # noqa: F401


@dataclass(frozen=True, slots=True)
class ImageSpanInfo:
    """One image placeholder span inside the prompt token stream."""

    start: int
    end: int
    num_encoder_tokens: int
    grid_thw: torch.Tensor | None


def span_end(mf: MultiModalFeature) -> int:
    """Canonical image-token end index (prefer length over stale ``end``)."""
    return int(mf.start + mf.num_encoder_tokens)


def overlaps_window(mf: MultiModalFeature, chunk_start: int, chunk_end: int) -> bool:
    return max(mf.start, chunk_start) < min(span_end(mf), chunk_end)


def build_mm_features(
    pixel_values: torch.Tensor,
    spans: list[ImageSpanInfo],
) -> list[MultiModalFeature]:
    """Slice HF ``pixel_values`` into per-image ``MultiModalFeature``s."""
    from ...scheduling.sequence import MultiModalFeature  # runtime import

    features: list[MultiModalFeature] = []
    offset = 0
    for i, span in enumerate(spans):
        g = span.grid_thw
        if g is not None and g.dim() == 1:
            g = g.view(1, -1)
        if g is None:
            n_patches = 0
            pv = pixel_values[offset:offset]
        else:
            n_patches = int(g.reshape(-1).prod().item())
            pv = pixel_values[offset : offset + n_patches].contiguous()
            offset += n_patches
        uid = hash_image_pixels(pv)
        features.append(
            MultiModalFeature(
                modality="image",
                data=pv.cpu(),
                start=span.start,
                end=span.end,
                num_encoder_tokens=span.num_encoder_tokens,
                uid=uid,
                grid_thw=g.cpu() if isinstance(g, torch.Tensor) else g,
            )
        )
    return features


def truncate_mm_features(
    token_ids: list[int],
    mm_features: list[MultiModalFeature],
    context_len: int,
) -> tuple[list[int], list[MultiModalFeature]]:
    """Left-truncate tokens; keep only fully-contained image spans."""
    keep_from = len(token_ids) - context_len
    kept: list[MultiModalFeature] = []
    for mf in mm_features:
        if mf.start >= keep_from and span_end(mf) <= len(token_ids):
            mf.start -= keep_from
            mf.end = mf.start + mf.num_encoder_tokens
            mf.mrope_origin = None
            kept.append(mf)
    return token_ids[-context_len:], kept
