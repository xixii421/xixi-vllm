import logging
from collections import OrderedDict

import torch
import torch.nn as nn

from .vit import Qwen3VisionTransformer

logger = logging.getLogger(__name__)


class EncoderCache:
    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self._cache: OrderedDict[str, tuple[torch.Tensor, list[torch.Tensor] | None]] = OrderedDict()

    def get(self, key: str):
        v = self._cache.get(key)
        if v is not None:
            self._cache.move_to_end(key)
        return v

    def put(self, key: str, embeds: torch.Tensor, deepstack=None):
        # Keep encoder outputs on GPU (official EncoderCacheManager semantics).
        ds = [t.detach() for t in deepstack] if deepstack is not None else None
        self._cache[key] = (embeds.detach(), ds)
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()

    def __contains__(self, k):
        return k in self._cache

    def __len__(self):
        return len(self._cache)


class Qwen3VisionTower(nn.Module):
    def __init__(self, vision_config) -> None:
        super().__init__()
        self._cfg = vision_config
        self.vit: Qwen3VisionTransformer | None = None

    @property
    def merge_size(self) -> int:
        return getattr(self._cfg, "spatial_merge_size", 2)

    @property
    def out_dim(self) -> int:
        return getattr(self._cfg, "out_hidden_size", self._cfg.hidden_size)

    @property
    def deepstack_features(self):
        return self.vit.deepstack_features if self.vit else None

    def load_weights(self, state_dict: dict[str, torch.Tensor], vision_config) -> None:
        self.vit = Qwen3VisionTransformer(vision_config).to(next(iter(state_dict.values())).device)
        n = self.vit.load_weights(state_dict)
        logger.info("Native ViT loaded: %d params", n)

    def prepare_metadata(self, grid_thw: torch.Tensor) -> dict:
        assert self.vit is not None
        return self.vit.prepare_metadata(grid_thw)

    def forward(self, pixel_values, grid_thw=None, metadata=None):
        assert self.vit is not None
        return self.vit(pixel_values, grid_thw=grid_thw, metadata=metadata)
