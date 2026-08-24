from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.model_loader import (  # noqa: E402
    default_weight_loader,
    load_model as _load_model_hf,
    safetensors_weights_iterator as _safetensors_weights_iterator,
)

from .sharded_loader import has_sharded_checkpoint, load_sharded_model
from .weight_utils import move_module

__all__ = [
    "default_weight_loader",
    "safetensors_weights_iterator",
    "load_model_hf",
    "load_model",
]


def safetensors_weights_iterator(
    hf_folder: str,
    device: str | torch.device | None = "cpu",
) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield weights from ``*.safetensors``; ``device`` is accepted for API compat."""
    del device
    return _safetensors_weights_iterator(hf_folder)


def load_model_hf(model: nn.Module, hf_folder: str) -> nn.Module:
    _load_model_hf(model, hf_folder)
    return model.eval()


def load_model(
    model: nn.Module,
    hf_folder: str,
    *,
    rank: int = 0,
    tp_size: int = 1,
    pp_size: int = 1,
    prefer_sharded: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """Load HF or rank-local shards, optionally placing onto ``device``/``dtype``."""
    if prefer_sharded and has_sharded_checkpoint(
        hf_folder,
        rank=rank,
        tp_size=tp_size,
        pp_size=pp_size,
    ):
        load_sharded_model(
            model,
            hf_folder,
            rank=rank,
            tp_size=tp_size,
            pp_size=pp_size,
        )
    else:
        load_model_hf(model, hf_folder)
    if device is not None and dtype is not None:
        return move_module(model, device=device, dtype=dtype)
    return model.eval()
