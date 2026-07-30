from __future__ import annotations

import glob
import os
from collections.abc import Iterable

import torch
from safetensors import safe_open
from torch import nn
from tqdm import tqdm


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    # 原地覆盖 Parameter 底层存储
    param.data.copy_(loaded_weight)


def safetensors_weights_iterator(hf_folder: str) -> Iterable[tuple[str, torch.Tensor]]:
    files = sorted(glob.glob(os.path.join(hf_folder, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {hf_folder}")
    for path in tqdm(files, desc="Loading weights"):
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def load_model(model: nn.Module, hf_folder: str) -> nn.Module:
    model.load_weights(safetensors_weights_iterator(hf_folder))
    return model.eval()
