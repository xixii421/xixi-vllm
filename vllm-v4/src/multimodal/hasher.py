from __future__ import annotations

import hashlib

import torch


def hash_image_pixels(pixel_values: torch.Tensor) -> str:
    """Stable content hash of the encoder input (official-style key).

    Hash the *encoded* pixel tensor (not the source PIL) so the cache key is
    automatically bound to image_processor config: config change → different
    pixel_values → different hash. No separate config fingerprint needed.
    """
    t = pixel_values.detach().contiguous()
    if t.device.type != "cpu":
        t = t.cpu()
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    if t.dtype == torch.bfloat16:
        raw = t.view(torch.int16).numpy().tobytes()
    else:
        raw = t.numpy().tobytes()
    h.update(raw)
    return h.hexdigest()
