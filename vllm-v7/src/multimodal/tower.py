from __future__ import annotations
import logging
from typing import Optional
import torch
import torch.nn as nn
from .vit import Qwen3VisionTransformer
logger = logging.getLogger(__name__)

class Qwen3VisionTower(nn.Module):

    def __init__(self, vision_config, *, disable_tp: bool=True) -> None:
        super().__init__()
        self._cfg = vision_config
        self.disable_tp = disable_tp
        self.vit: Optional[Qwen3VisionTransformer] = None

    @property
    def merge_size(self) -> int:
        return getattr(self._cfg, 'spatial_merge_size', 2)

    @property
    def out_dim(self) -> int:
        return getattr(self._cfg, 'out_hidden_size', self._cfg.hidden_size)

    @property
    def deepstack_features(self) -> Optional[list[torch.Tensor]]:
        return self.vit.deepstack_features if self.vit else None

    def load_weights(self, state_dict: dict[str, torch.Tensor], vision_config) -> None:
        self.vit = Qwen3VisionTransformer(vision_config, disable_tp=self.disable_tp).to(device=next(iter(state_dict.values())).device)
        n = self.vit.load_weights(state_dict)
        logger.info('Native ViT loaded: %s params (disable_tp=%s)', n, self.disable_tp)

    def prepare_metadata(self, grid_thw: torch.Tensor) -> dict:
        return self.vit.prepare_metadata(grid_thw)

    def forward(self, pixel_values: torch.Tensor, grid_thw: Optional[torch.Tensor]=None, metadata: Optional[dict]=None, out_merged: Optional[torch.Tensor]=None, out_deepstack: Optional[list[torch.Tensor]]=None) -> tuple:
        return self.vit(pixel_values, grid_thw=grid_thw, metadata=metadata, out_merged=out_merged, out_deepstack=out_deepstack)
