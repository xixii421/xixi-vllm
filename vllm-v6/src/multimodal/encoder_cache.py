from __future__ import annotations
from collections import OrderedDict
from typing import Optional
import torch

class EncoderCache:

    def __init__(self, max_size: int=256):
        self.max_size = max_size
        self._cache: OrderedDict[str, tuple[torch.Tensor, Optional[list[torch.Tensor]]]] = OrderedDict()

    def get(self, key: str) -> Optional[tuple[torch.Tensor, Optional[list[torch.Tensor]]]]:
        v = self._cache.get(key)
        if v is not None:
            self._cache.move_to_end(key)
        return v

    def put(self, key: str, embeds: torch.Tensor, deepstack: Optional[list[torch.Tensor]]=None):
        ds = None
        if deepstack is not None:
            ds = [t.detach().to('cpu') for t in deepstack]
        payload = (embeds.detach(), ds)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = payload
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = payload

    def clear(self):
        self._cache.clear()

    def __contains__(self, k):
        return k in self._cache

    def __len__(self):
        return len(self._cache)
