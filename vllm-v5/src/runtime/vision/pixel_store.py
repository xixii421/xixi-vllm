"""Worker-side pixel registry keyed by multimodal ``uid``.

Engine sends pixels once via ``register_mm_pixels``; encode looks up by uid
when ``MultiModalFeature.data`` is absent on the RPC wire.
"""

from __future__ import annotations

import torch


class PixelStore:
    def __init__(self) -> None:
        self._by_uid: dict[str, torch.Tensor] = {}

    def put_many(self, items: dict[str, torch.Tensor]) -> None:
        for uid, data in items.items():
            self._by_uid[uid] = data

    def get(self, uid: str) -> torch.Tensor | None:
        return self._by_uid.get(uid)

    def discard(self, uid: str) -> None:
        self._by_uid.pop(uid, None)
