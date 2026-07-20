from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ViTCGConfig:
    min_budget: int = 64
    max_budget: int = 256
    out_hidden_size: int = 2048
    merge_size: int = 2
    patch_dim: int = 1536
    base_grid: tuple[int, int, int] = (1, 16, 16)
    num_deepstack: int = 0
