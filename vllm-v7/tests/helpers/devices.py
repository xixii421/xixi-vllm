"""GPU pick helpers for MoE / parallel tests.

Engine rules (see ``EngineConfig._validate_tp_homogeneous_devices``):
- **TP / EP** (same process group): require same-SKU GPUs unless
  ``allow_hetero_tp=True``.
- **PP** across stages: heterogeneous SKUs are allowed (recommended for
  Blackwell + A100 → ``tp=1, pp=2, device_ids=[bw, a100]``).
"""

from __future__ import annotations

import os

import torch


def gpu_name(i: int) -> str:
    return torch.cuda.get_device_properties(i).name


def gpu_mem_mib(i: int) -> int:
    return int(torch.cuda.get_device_properties(i).total_memory // (1024 * 1024))


def gpu_free_mib(i: int) -> int:
    try:
        free, _ = torch.cuda.mem_get_info(i)
        return int(free // (1024 * 1024))
    except Exception:
        return gpu_mem_mib(i)


def find_gpu(*needles: str, min_free_mib: int = 0) -> int | None:
    """First GPU whose name contains any needle (case-insensitive)."""
    if not torch.cuda.is_available():
        return None
    lowered = [n.lower() for n in needles]
    for i in range(torch.cuda.device_count()):
        name = gpu_name(i).lower()
        if any(n in name for n in lowered):
            if min_free_mib and gpu_free_mib(i) < min_free_mib:
                continue
            return i
    return None


def find_blackwell(*, min_free_mib: int = 0) -> int | None:
    # Prefer explicit Blackwell / RTX PRO naming (avoid bare "6000" false positives).
    return find_gpu("blackwell", "rtx pro 6000", "pro 6000", min_free_mib=min_free_mib)


def find_a100(*, min_free_mib: int = 0) -> int | None:
    return find_gpu("a100", min_free_mib=min_free_mib)


def blackwell_a100_pair(
    *,
    min_free_mib: int = 0,
    order: str = "bw_a100",
) -> list[int] | None:
    """Return ``[stage0, stage1]`` for hetero PP (default Blackwell → A100)."""
    bw = find_blackwell(min_free_mib=min_free_mib)
    a100 = find_a100(min_free_mib=min_free_mib)
    if bw is None or a100 is None:
        return None
    if order == "a100_bw":
        return [a100, bw]
    return [bw, a100]


def env_device_ids(key: str = "VLLM_V7_MOE_DEVICES") -> list[int] | None:
    """Optional override, e.g. ``VLLM_V7_MOE_DEVICES=0,3``."""
    raw = os.environ.get(key, "").strip()
    if not raw:
        return None
    return [int(x) for x in raw.split(",") if x.strip() != ""]
