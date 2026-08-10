from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download


DEFAULT_CACHE_DIR = Path(
    os.environ.get("HF_HOME")
    or os.environ.get("HUGGINGFACE_HUB_HOME")
    or Path.home() / "huggingface"
) # 默认缓存目录

@dataclass(frozen=True)
class ModelSpec:
    """注册表条目：本地目录名、HF repo、约估计的 fp16 体积（GB）。"""
    local_name: str
    repo_id: str # HuggingFace 线上仓库id
    fp16_gb: float

QWEN3_REGISTRY: tuple[ModelSpec,...] = (
    ModelSpec("Qwen3-0.6B","Qwen/Qwen3-0.6B",1.2),
    ModelSpec("Qwen3-1.7B", "Qwen/Qwen3-1.7B", 3.4),
) # 模型注册表元组，存放模型信息

def is_downloaded(spec:ModelSpec,cache_dir:Path | None=None) -> bool:
    path = (cache_dir or DEFAULT_CACHE_DIR) / spec.local_name
    # 如果传入缓存目录就用，不传入就用默认目录，之后拼接文件夹名
    return path.is_dir() and any(path.glob("*.safetensors"))
    # 满足两个条件才能确认下载完成 1 路径必须是真实存在的文件夹 2 文件夹内至少存在一个后缀为.safetensors的权重文件

def download_model(spec:ModelSpec,cache_dir:Path | None=None) -> Path:
    root = cache_dir or DEFAULT_CACHE_DIR
    target = root / spec.local_name
    if is_downloaded(spec,root):
        print(f"[skip] {spec.local_name} already at {target}")
        return target
    snapshot_download(
        spec.repo_id,
        local_dir=str(target), # 下载路径，snapshot_download 接收字符串路径
        token=os.environ.get("HF_TOKEN"), # 读取环境变量里的HF访问令牌，公开模型不需要
        endpoint=os.environ.get("HF_ENDPOINT") or None, # HF服务地址
    ) # 执行下载
    return target

def download_models(
    specs: list[ModelSpec] | None = None,
    cache_dir: Path | None = None,
) -> list[Path]:
    return [download_model(s,cache_dir) for s in (specs or list(QWEN3_REGISTRY))]


def gpu_memory_gb() -> float: # 获取0号GPU显卡的总物理显存大小
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    # 获取第0号gpu的硬件信息 total_memory 显卡总显存，单位是字节

def select_models_for_vram(
    vram_gb: float | None = None,
    reserved_gb: float = 4.0,
    registry: tuple[ModelSpec,...] = QWEN3_REGISTRY,
) -> list[ModelSpec]:
    """按「单模型可装下」粗筛；校验时仍是一个一个加载。"""
    budget = (vram_gb if vram_gb is not None else gpu_memory_gb() or 24.0) - reserved_gb
    # 1 vram_gb 手动指定显卡总显存，不传则自动调用gpu_memory_gb()读取硬件显存
    # 2 reserved_gb 预留显存，默认4GB
    return [s for s in registry if s.fp16_gb <= budget] # 筛选符合条件的模型

