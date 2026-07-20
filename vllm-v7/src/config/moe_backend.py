"""MoE fused 后端选择：FlashInfer cutlass（优先）或 Triton。

cutlass：SM 89/90/100+；SM120 需系统 nvcc >= 12.9（JIT ``120f``）。
triton：覆盖 Ampere（A100/3090）与 cutlass 不可用时的消费级卡。
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Literal

MoeBackend = Literal["cutlass", "triton"]

_CUTLASS_SMS = frozenset({89, 90, 100, 103, 110, 120, 121})


def _sm_tag(device: int | None = None) -> int | None:
    import torch

    if not torch.cuda.is_available():
        return None
    if device is None:
        major, minor = torch.cuda.get_device_capability()
    else:
        major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


def _nvcc_version() -> tuple[int, int] | None:
    candidates: list[str] = []
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if home:
        candidates.append(os.path.join(home, "bin", "nvcc"))
    candidates.append("nvcc")
    for nvcc in candidates:
        try:
            out = subprocess.check_output(
                [nvcc, "--version"], stderr=subprocess.STDOUT, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError):
            continue
        m = re.search(r"release\s+(\d+)\.(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def _flashinfer_cutlass_importable() -> bool:
    try:
        from flashinfer import cutlass_fused_moe  # noqa: F401

        return True
    except ImportError:
        return False


def cutlass_moe_available(*, device: int | None = None) -> bool:
    sm = _sm_tag(device)
    if sm is None or sm not in _CUTLASS_SMS:
        return False
    if not _flashinfer_cutlass_importable():
        return False
    if sm >= 120:
        ver = _nvcc_version()
        # Full ``12.0f`` path needs nvcc ≥ 12.9. With nvcc 12.8 we still allow
        # cutlass via FlashInfer arch ``12.0a`` (see format_flashinfer_arch).
        if ver is None:
            return False
        if ver < (12, 8):
            return False
    return True


def triton_moe_available() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_moe_backend(
    prefer: Literal["auto", "cutlass", "triton"] = "auto",
    *,
    device: int | None = None,
) -> MoeBackend:
    """解析本进程应使用的 MoE 内核。"""
    if prefer == "cutlass":
        if not cutlass_moe_available(device=device):
            raise RuntimeError(
                "moe backend=cutlass requested but unavailable "
                f"(sm={_sm_tag(device)} nvcc={_nvcc_version()})"
            )
        return "cutlass"
    if prefer == "triton":
        if not triton_moe_available():
            raise RuntimeError("moe backend=triton requested but triton is not installed")
        return "triton"
    # auto
    if cutlass_moe_available(device=device):
        return "cutlass"
    if triton_moe_available():
        return "triton"
    raise RuntimeError(
        "No MoE backend available: need FlashInfer cutlass (SM89+/toolkit) "
        "or Triton. "
        f"sm={_sm_tag(device)} nvcc={_nvcc_version()}"
    )


def require_moe_backend(
    prefer: Literal["auto", "cutlass", "triton"] = "auto",
    *,
    device: int | None = None,
) -> MoeBackend:
    return resolve_moe_backend(prefer, device=device)


# 兼容旧名
def require_cutlass_moe(*, device: int | None = None) -> None:
    if not cutlass_moe_available(device=device):
        raise RuntimeError(
            "FlashInfer cutlass MoE unavailable on this device; "
            "use moe_fused_backend='auto'/'triton' or install CUDA toolkit>=12.9 for SM120."
        )


def moe_decode_cg_ok(
    backend: MoeBackend | None = None,
    *,
    device: int | None = None,
) -> bool:
    """cutlass 适合 CUDA Graph；精简 Triton 路径默认关 CG（更稳）。"""
    if backend is None:
        try:
            backend = resolve_moe_backend("auto", device=device)
        except RuntimeError:
            return False
    return backend == "cutlass"
