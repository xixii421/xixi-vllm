from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from .vit import Qwen3VisionTransformer

logger = logging.getLogger(__name__)


class Qwen3VisionTower(nn.Module):
    def __init__(self, vision_config, *, use_weight_tp: bool = False) -> None:
        super().__init__()
        self._cfg = vision_config
        self.use_weight_tp = use_weight_tp
        self.vit: Optional[Qwen3VisionTransformer] = None

    @property
    def merge_size(self) -> int:
        return int(getattr(self._cfg, "spatial_merge_size", 2))

    @property
    def out_dim(self) -> int:
        return int(getattr(self._cfg, "out_hidden_size", self._cfg.hidden_size))

    @property
    def deepstack_features(self) -> Optional[list[torch.Tensor]]:
        return None if self.vit is None else self.vit.deepstack_features

    def load_weights(self, state_dict: dict[str, torch.Tensor], vision_config) -> None:
        device = next(iter(state_dict.values())).device
        self.vit = Qwen3VisionTransformer(
            vision_config, use_weight_tp=self.use_weight_tp
        ).to(device=device)
        n = self.vit.load_weights(state_dict)
        mode = "weight-TP" if self.use_weight_tp else "replicated"
        logger.info("Native ViT loaded (%s): %s params", mode, n)

    def prepare_metadata(self, grid_thw: torch.Tensor) -> dict:
        assert self.vit is not None
        return self.vit.prepare_metadata(grid_thw)

    def forward(self, *args, **kwargs):
        assert self.vit is not None, "vision tower weights not loaded"
        return self.vit(*args, **kwargs)
