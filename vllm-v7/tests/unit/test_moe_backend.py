"""MoE backend resolve: cutlass prefer, triton fallback."""

from __future__ import annotations

import pytest

from src.config.moe_backend import (
    cutlass_moe_available,
    moe_decode_cg_ok,
    require_moe_backend,
    resolve_moe_backend,
    triton_moe_available,
)


def test_resolve_auto_returns_known(monkeypatch):
    monkeypatch.setattr(
        "src.config.moe_backend.cutlass_moe_available", lambda **k: False
    )
    monkeypatch.setattr("src.config.moe_backend.triton_moe_available", lambda: True)
    assert resolve_moe_backend("auto") == "triton"


def test_prefer_cutlass_raises(monkeypatch):
    monkeypatch.setattr(
        "src.config.moe_backend.cutlass_moe_available", lambda **k: False
    )
    with pytest.raises(RuntimeError, match="cutlass"):
        require_moe_backend("cutlass")


def test_cg_only_cutlass():
    assert moe_decode_cg_ok("cutlass") is True
    assert moe_decode_cg_ok("triton") is False


def test_sm120_allows_nvcc_128_via_arch_a(monkeypatch):
    """SM120 + nvcc 12.8 → cutlass OK (FlashInfer arch 12.0a); <12.8 → no."""
    monkeypatch.setattr(
        "src.config.moe_backend._sm_tag", lambda device=None: 120
    )
    monkeypatch.setattr(
        "src.config.moe_backend._flashinfer_cutlass_importable", lambda: True
    )
    monkeypatch.setattr(
        "src.config.moe_backend._nvcc_version", lambda: (12, 7)
    )
    assert cutlass_moe_available() is False
    monkeypatch.setattr(
        "src.config.moe_backend._nvcc_version", lambda: (12, 8)
    )
    assert cutlass_moe_available() is True
    monkeypatch.setattr(
        "src.config.moe_backend._nvcc_version", lambda: (12, 9)
    )
    assert cutlass_moe_available() is True


def test_triton_package_present():
    # 教学环境应装 triton；未装则跳过而非 fail CI 逻辑外的断言
    assert isinstance(triton_moe_available(), bool)
