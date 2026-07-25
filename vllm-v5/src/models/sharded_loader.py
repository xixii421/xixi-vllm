from __future__ import annotations

import glob
import json
import os
from collections.abc import Iterable
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from tqdm import tqdm

from .weight_utils import default_weight_loader

SHARD_PATTERN = "model-rank-{rank}-part-{part}.safetensors"
MANIFEST_NAME = "vllm_v7_sharded_manifest.json"
DEFAULT_MAX_PART_BYTES = 4 * 1024**3


def sharded_root(model_dir: str, *, tp_size: int, pp_size: int) -> str:
    return os.path.join(model_dir, "sharded", f"tp{tp_size}_pp{pp_size}")


def _manifest_path(model_dir: str, *, tp_size: int, pp_size: int) -> str:
    return os.path.join(sharded_root(model_dir, tp_size=tp_size, pp_size=pp_size), MANIFEST_NAME)


def _shard_glob(model_dir: str, *, rank: int, tp_size: int, pp_size: int) -> str:
    root = sharded_root(model_dir, tp_size=tp_size, pp_size=pp_size)
    return os.path.join(root, SHARD_PATTERN.format(rank=rank, part="*"))


def read_manifest(model_dir: str, *, tp_size: int, pp_size: int) -> dict[str, Any] | None:
    path = _manifest_path(model_dir, tp_size=tp_size, pp_size=pp_size)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_manifest(
    model_dir: str,
    *,
    tp_size: int,
    pp_size: int,
    world_size: int,
    dtype: str | None = None,
) -> None:
    root = sharded_root(model_dir, tp_size=tp_size, pp_size=pp_size)
    os.makedirs(root, exist_ok=True)
    payload = {
        "format": "vllm-v7-sharded",
        "tp_size": int(tp_size),
        "pp_size": int(pp_size),
        "world_size": int(world_size),
        "pattern": SHARD_PATTERN,
        "root": os.path.relpath(root, model_dir),
    }
    if dtype is not None:
        payload["dtype"] = dtype
    with open(_manifest_path(model_dir, tp_size=tp_size, pp_size=pp_size), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def has_sharded_checkpoint(
    model_dir: str,
    *,
    rank: int,
    tp_size: int,
    pp_size: int,
) -> bool:
    manifest = read_manifest(model_dir, tp_size=tp_size, pp_size=pp_size)
    if manifest is None:
        return False
    if int(manifest.get("tp_size", -1)) != tp_size:
        return False
    if int(manifest.get("pp_size", -1)) != pp_size:
        return False
    if int(manifest.get("world_size", -1)) != tp_size * pp_size:
        return False
    return bool(glob.glob(_shard_glob(model_dir, rank=rank, tp_size=tp_size, pp_size=pp_size)))


def iter_sharded_weights(
    model_dir: str,
    *,
    rank: int,
    tp_size: int,
    pp_size: int,
) -> Iterable[tuple[str, torch.Tensor]]:
    files = sorted(glob.glob(_shard_glob(model_dir, rank=rank, tp_size=tp_size, pp_size=pp_size)))
    if not files:
        raise FileNotFoundError(
            f"No sharded checkpoint files for rank={rank} under "
            f"{sharded_root(model_dir, tp_size=tp_size, pp_size=pp_size)}"
        )
    for shard_file in tqdm(files, desc=f"Loading rank-{rank} shards"):
        with safe_open(shard_file, framework="pt") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def load_sharded_model(
    model: nn.Module,
    model_dir: str,
    *,
    rank: int,
    tp_size: int,
    pp_size: int,
) -> set[str]:
    params = dict(model.named_parameters())
    loaded: set[str] = set()
    for name, tensor in iter_sharded_weights(
        model_dir,
        rank=rank,
        tp_size=tp_size,
        pp_size=pp_size,
    ):
        if name not in params:
            continue
        default_weight_loader(params[name], tensor)
        loaded.add(name)
    missing = [name for name in params if name not in loaded]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise RuntimeError(
            f"Sharded checkpoint for rank={rank} is missing {len(missing)} parameters: "
            f"{preview}{suffix}"
        )
    return loaded


def save_sharded_checkpoint(
    model: nn.Module,
    model_dir: str,
    *,
    rank: int,
    tp_size: int,
    pp_size: int,
    max_part_bytes: int = DEFAULT_MAX_PART_BYTES,
    dtype: torch.dtype | None = None,
) -> list[str]:
    """Persist rank-local parameters.

    When ``dtype`` is set, tensors are cast before writing so dense/MoE shards
    match the runtime dtype (same as the VL save path).
    """
    root = sharded_root(model_dir, tp_size=tp_size, pp_size=pp_size)
    os.makedirs(root, exist_ok=True)
    written: list[str] = []
    part = 0
    bucket: dict[str, torch.Tensor] = {}
    bucket_bytes = 0

    def flush() -> None:
        nonlocal part, bucket, bucket_bytes
        if not bucket:
            return
        out = os.path.join(root, SHARD_PATTERN.format(rank=rank, part=part))
        save_file(bucket, out)
        written.append(out)
        part += 1
        bucket = {}
        bucket_bytes = 0

    for name, param in model.named_parameters():
        tensor = param.detach().cpu().contiguous()
        if dtype is not None and tensor.is_floating_point() and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        nbytes = tensor.numel() * tensor.element_size()
        if bucket and bucket_bytes + nbytes > max_part_bytes:
            flush()
        bucket[name] = tensor
        bucket_bytes += nbytes
    flush()
    if not written:
        raise RuntimeError(f"No parameters to save for rank={rank}")
    return written
