"""Local HF model paths used by GPU / distributed tests."""

from __future__ import annotations

from pathlib import Path

from src.utils.paths import resolve_hf_home

HF_HOME = resolve_hf_home()


def _complete_model(*candidates: Path) -> Path:
    """First directory that contains at least one ``*.safetensors`` weight file."""
    for p in candidates:
        if p.is_dir() and any(p.glob("*.safetensors")):
            return p
    return candidates[0]


TEXT_MODEL = _complete_model(HF_HOME / "Qwen3-0.6B", HF_HOME / "Qwen3-8B")
VL_MODEL = _complete_model(
    HF_HOME / "Qwen3-VL-2B-Instruct",
    HF_HOME / "Qwen3-VL-8B-Instruct",
)
# 优先较小真实 MoE（~14B / 激活 2.7B）；否则回退 30B-A3B
MOE_MODEL = _complete_model(
    HF_HOME / "Qwen1.5-MoE-A2.7B",
    HF_HOME / "Qwen1.5-MoE-A2.7B-Chat",
    HF_HOME / "Qwen3-30B-A3B",
    HF_HOME / "Qwen3-30B-A3B-Instruct",
)


def require_model(path: Path):
    """Skip decorator when a local model directory is missing or incomplete."""
    import pytest

    ok = path.is_dir() and any(path.glob("*.safetensors"))
    return pytest.mark.skipif(not ok, reason=f"missing/incomplete model: {path}")
