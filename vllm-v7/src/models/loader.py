from __future__ import annotations

import glob
import os
from collections.abc import Iterable

import torch
from safetensors import safe_open
from torch import nn
from tqdm import tqdm

from .sharded_loader import has_sharded_checkpoint, load_sharded_model
from .weight_utils import default_weight_loader, move_module


def safetensors_weights_iterator(hf_folder: str) -> Iterable[tuple[str, torch.Tensor]]:
    files = sorted(glob.glob(os.path.join(hf_folder, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {hf_folder}")
    for st_file in tqdm(files, desc="Loading weights"):
        with safe_open(st_file, framework="pt") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def load_model_hf(model: nn.Module, hf_folder: str) -> nn.Module:
    model.load_weights(safetensors_weights_iterator(hf_folder))
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
