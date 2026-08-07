from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

# 缓存根目录：优先环境变量，其次退回 ~/huggingface
DEFAULT_CACHE_DIR = Path(
    os.environ.get("HF_HOME")
    or os.environ.get("HUGGINGFACE_HUB_CACHE")
    or Path.home() / "huggingface"
)


@dataclass(frozen=True)
class ModelSpec:
    """注册表条目：本地目录名、HF repo、约估计的 fp16 体积（GB）。"""

    local_name: str
    repo_id: str
    fp16_gb: float


QWEN3_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec("Qwen3-0.6B", "Qwen/Qwen3-0.6B", 1.2),
    ModelSpec("Qwen3-1.7B", "Qwen/Qwen3-1.7B", 3.4),
    ModelSpec("Qwen3-4B", "Qwen/Qwen3-4B", 8.0),
)


def is_downloaded(spec: ModelSpec, cache_dir: Path | None = None) -> bool:
    path = (cache_dir or DEFAULT_CACHE_DIR) / spec.local_name
    return path.is_dir() and any(path.glob("*.safetensors"))


def download_model(spec: ModelSpec, cache_dir: Path | None = None) -> Path:
    root = cache_dir or DEFAULT_CACHE_DIR
    target = root / spec.local_name
    if is_downloaded(spec, root):
        print(f"[skip] {spec.local_name} already at {target}")
        return target
    print(f"[download] {spec.repo_id} -> {target}")
    snapshot_download(
        spec.repo_id,
        local_dir=str(target),
        token=os.environ.get("HF_TOKEN"),
        endpoint=os.environ.get("HF_ENDPOINT") or None,
    )
    return target


def download_models(
    specs: list[ModelSpec] | None = None,
    cache_dir: Path | None = None,
) -> list[Path]:
    return [download_model(s, cache_dir) for s in (specs or list(QWEN3_REGISTRY))]


def gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def select_models_for_vram(
    vram_gb: float | None = None,
    reserve_gb: float = 4.0,
    registry: tuple[ModelSpec, ...] = QWEN3_REGISTRY,
) -> list[ModelSpec]:
    """按「单模型可装下」粗筛；校验时仍是一个一个加载。"""
    budget = (vram_gb if vram_gb is not None else gpu_memory_gb() or 24.0) - reserve_gb
    return [s for s in registry if s.fp16_gb <= budget]
