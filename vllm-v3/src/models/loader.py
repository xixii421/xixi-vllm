from __future__ import annotations

import concurrent.futures
import glob
import os
from collections.abc import Iterable
from contextlib import contextmanager

import torch
from safetensors.torch import load_file
from torch import nn
from tqdm import tqdm


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


@contextmanager
def skip_param_init():
    """跳过模块构造时的随机参数初始化。

    权重随后由 load_model 覆盖，先分配随机数是纯浪费——4B 模型约 24s。
    临时替换 reset_parameters，构建完自动恢复。
    """
    orig_linear = nn.Linear.reset_parameters
    orig_embed = nn.Embedding.reset_parameters
    nn.Linear.reset_parameters = lambda self: None
    nn.Embedding.reset_parameters = lambda self: None
    try:
        yield
    finally:
        nn.Linear.reset_parameters = orig_linear
        nn.Embedding.reset_parameters = orig_embed


@contextmanager
def default_dtype_context(dtype: torch.dtype):
    """在目标 dtype 下构建模型，省去 fp32 中间态（显存减半、无 dtype 转换）。"""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def _load_shard_cpu(path: str) -> list[tuple[str, torch.Tensor]]:
    return list(load_file(path, device="cpu").items())


def safetensors_weights_iterator(hf_folder: str) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` from ``*.safetensors`` under ``hf_folder``.

    整分片读入（load_file）并后台预取下一分片，磁盘 I/O 与 CPU copy 重叠。
    """
    files = sorted(glob.glob(os.path.join(hf_folder, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {hf_folder}")

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


def load_model(model: nn.Module, hf_folder: str) -> nn.Module:
    model.load_weights(safetensors_weights_iterator(hf_folder))
    return model.eval()
