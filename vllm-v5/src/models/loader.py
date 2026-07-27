from __future__ import annotations

import concurrent.futures
import glob
import os
from collections.abc import Iterable

import torch
from safetensors.torch import load_file
from torch import nn
from tqdm import tqdm

from .sharded_loader import has_sharded_checkpoint, load_sharded_model
from .weight_utils import default_weight_loader, move_module

__all__ = [
    "default_weight_loader",
    "safetensors_weights_iterator",
    "load_model_hf",
    "load_model",
]


def _load_shard_cpu(path: str) -> list[tuple[str, torch.Tensor]]:
    return list(load_file(path, device="cpu").items())


def safetensors_weights_iterator(
    hf_folder: str,
    device: str | torch.device | None = "cpu",
) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` from ``*.safetensors`` under ``hf_folder``.

    Always materializes on CPU via ``load_file``. ``device`` is accepted for API
    compatibility with callers that pass a target device, but is ignored — the
    existing ``param.copy_`` path expects host tensors.
    """
    del device
    files = glob.glob(os.path.join(hf_folder, "*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {hf_folder}")
    files.sort()

    if len(files) == 1:
        for st_file in tqdm(files, desc="Loading weights"):
            yield from _load_shard_cpu(st_file)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_load_shard_cpu, files[0])
        for i, _st_file in enumerate(tqdm(files, desc="Loading weights")):
            items = fut.result()
            if i + 1 < len(files):
                fut = pool.submit(_load_shard_cpu, files[i + 1])
            yield from items


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
