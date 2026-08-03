from __future__ import annotations

import glob
import os
from collections.abc import Iterable

import torch
from safetensors import safe_open
from torch import nn
from tqdm import tqdm


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight)


def safetensors_weights_iterator(hf_folder: str) -> Iterable[tuple[str, torch.Tensor]]:
    # 生成器,把权重目录流式变成(name, tensor)对, 一次只持有一个张量,避免一次性把整个模型读进内存
    files = sorted(glob.glob(os.path.join(hf_folder, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {hf_folder}")
    for path in tqdm(files, desc="Loading weights"):
        with safe_open(path, framework="pt") as f:
            # get_tensor 惰性读取, yield后暂停,等到model.load_weights处理完再继续
            for name in f.keys():
                yield name, f.get_tensor(name)


def load_model(model: nn.Module, hf_folder: str) -> nn.Module:
    # 管道入口, 创建生成器
    model.load_weights(safetensors_weights_iterator(hf_folder))
    return model
