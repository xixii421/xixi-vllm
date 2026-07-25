"""FlashInfer arch pinning (mixed-GPU hosts)."""

from __future__ import annotations

import os

import pytest

from src.utils.cuda_env import (
    apply_flashinfer_arch_list,
    format_flashinfer_arch,
    flashinfer_arch_list_for_devices,
)


def test_format_flashinfer_arch_ampere_ada():
    assert format_flashinfer_arch(8, 0) == "8.0"
    assert format_flashinfer_arch(8, 6) == "8.6"
    assert format_flashinfer_arch(9, 0) == "9.0a"


def test_format_flashinfer_arch_sm12_uses_nvcc(monkeypatch):
    # Without nvcc ≥12.9, SM12 must emit 12.0a (works with toolkit 12.8).
    monkeypatch.setenv("CUDA_HOME", "/nonexistent")
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()),
    )
    assert format_flashinfer_arch(12, 0) == "12.0a"


@pytest.mark.skipif(
    __import__("torch").cuda.device_count() < 1,
    reason="needs at least one CUDA device",
)
def test_apply_flashinfer_arch_list_pins_selected_devices(monkeypatch):
    import torch

    monkeypatch.delenv("FLASHINFER_CUDA_ARCH_LIST", raising=False)
    # Prefer a non-zero device when available so we exercise "ignore GPU0".
    ids = [1] if torch.cuda.device_count() > 1 else [0]
    arch = apply_flashinfer_arch_list(ids, overwrite=True)
    assert os.environ["FLASHINFER_CUDA_ARCH_LIST"] == arch
    major, minor = torch.cuda.get_device_capability(ids[0])
    assert format_flashinfer_arch(major, minor) in arch.split()
    # Autodetect must not leave an empty TARGET after refresh.
    from flashinfer.jit.core import check_cuda_arch, current_compilation_context

    assert current_compilation_context.TARGET_CUDA_ARCHS
    check_cuda_arch()


@pytest.mark.skipif(
    __import__("torch").cuda.device_count() < 2,
    reason="needs >=2 GPUs",
)
def test_arch_list_only_includes_requested_devices(monkeypatch):
    import torch

    monkeypatch.delenv("FLASHINFER_CUDA_ARCH_LIST", raising=True)
    # Device 1 only — must not require scanning / including device 0's arch.
    arch = flashinfer_arch_list_for_devices([1])
    major, minor = torch.cuda.get_device_capability(1)
    assert arch == format_flashinfer_arch(major, minor)
