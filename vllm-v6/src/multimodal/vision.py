from typing import Optional
from collections import OrderedDict
import torch, torch.nn as nn
from .vit import Qwen3VisionTransformer
import logging
logger = logging.getLogger(__name__)


class EncoderCache:

    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self._cache: OrderedDict[str, tuple[torch.Tensor, Optional[list[torch.Tensor]]]] = (
            OrderedDict()
        )

    def get(self, key: str) -> Optional[tuple[torch.Tensor, Optional[list[torch.Tensor]]]]:
        v = self._cache.get(key)
        if v is not None:
            self._cache.move_to_end(key)
        return v

    def put(
        self,
        key: str,
        embeds: torch.Tensor,
        deepstack: Optional[list[torch.Tensor]] = None,
    ):
        ds = None
        if deepstack is not None:
            ds = [t.detach().to("cpu") for t in deepstack]
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


class Qwen3VisionTower(nn.Module):

    def __init__(self, vision_config) -> None:
        super().__init__()
        self._cfg = vision_config
        self.vit: Optional[Qwen3VisionTransformer] = None

    @property
    def merge_size(self) -> int:
        return getattr(self._cfg, "spatial_merge_size", 2)

    @property
    def out_dim(self) -> int:
        return getattr(self._cfg, "out_hidden_size", self._cfg.hidden_size)

    @property
    def deepstack_features(self) -> Optional[list[torch.Tensor]]:
        return self.vit.deepstack_features if self.vit else None

    def load_weights(self, state_dict: dict[str, torch.Tensor], vision_config) -> None:
        """从 HF state_dict 创建并加载 ViT。"""
        self.vit = Qwen3VisionTransformer(vision_config).to(
            device=next(iter(state_dict.values())).device)
        n = self.vit.load_weights(state_dict)
        logger.info(f"Native ViT loaded: {n} params")

    def prepare_metadata(self, grid_thw: torch.Tensor) -> dict:
        return self.vit.prepare_metadata(grid_thw)

    def forward(self, pixel_values: torch.Tensor,
                grid_thw: Optional[torch.Tensor] = None,
                metadata: Optional[dict] = None,
                out_merged: Optional[torch.Tensor] = None,
                out_deepstack: Optional[list[torch.Tensor]] = None) -> tuple:
        return self.vit(
            pixel_values,
            grid_thw=grid_thw,
            metadata=metadata,
            out_merged=out_merged,
            out_deepstack=out_deepstack,
        )
