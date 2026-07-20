"""Worker-local multimodal encoder embedding cache."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch


class EncoderCache:
    def __init__(
        self,
        max_size: int = 256,
        *,
        max_bytes: int | None = 512 * 1024 * 1024,
    ) -> None:
        self.max_size = max_size
        self.max_bytes = max_bytes
        self._cache: OrderedDict[
            str, tuple[torch.Tensor, Optional[list[torch.Tensor]], int]
        ] = OrderedDict()
        self._nbytes = 0

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def _entry_nbytes(
        self, embeds: torch.Tensor, deepstack: Optional[list[torch.Tensor]]
    ) -> int:
        n = embeds.numel() * embeds.element_size()
        if deepstack:
            for t in deepstack:
                n += t.numel() * t.element_size()
        return n

    def get(
        self, key: str
    ) -> Optional[tuple[torch.Tensor, Optional[list[torch.Tensor]]]]:
        v = self._cache.get(key)
        if v is None:
            return None
        self._cache.move_to_end(key)
        emb, ds, _ = v
        return emb, ds

    def put(
        self,
        key: str,
        embeds: torch.Tensor,
        deepstack: Optional[list[torch.Tensor]] = None,
    ) -> None:
        emb = embeds.detach().contiguous().to("cpu")
        ds = None
        if deepstack is not None:
            ds = [t.detach().contiguous().to("cpu") for t in deepstack]
        nbytes = self._entry_nbytes(emb, ds)
        if key in self._cache:
            _, _, old_n = self._cache.pop(key)
            self._nbytes -= old_n
        while self._cache and (
            len(self._cache) >= self.max_size
            or (self.max_bytes is not None and self._nbytes + nbytes > self.max_bytes)
        ):
            _, (_, _, old_n) = self._cache.popitem(last=False)
            self._nbytes -= old_n
        self._cache[key] = (emb, ds, nbytes)
        self._nbytes += nbytes

    def clear(self) -> None:
        self._cache.clear()
        self._nbytes = 0

    def __contains__(self, k: str) -> bool:
        return k in self._cache

    def __len__(self) -> int:
        return len(self._cache)
