"""Stable content hashes for multimodal encoder-cache keys."""

from __future__ import annotations
import hashlib
import torch


def hash_image_pixels(pixel_values: torch.Tensor) -> str:
    t = pixel_values.detach().contiguous()
    if t.device.type != "cpu":
        t = t.cpu()
    # Prefer a compact dtype for hashing without changing values used for encode.
    if t.dtype == torch.bfloat16:
        raw = t.view(torch.int16).numpy().tobytes()
    else:
        raw = t.numpy().tobytes()
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    h.update(raw)
    return h.hexdigest()
