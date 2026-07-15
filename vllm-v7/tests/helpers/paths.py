"""Local HF model paths used by GPU / distributed tests."""

from __future__ import annotations

from pathlib import Path

import pytest

HF_HOME = Path.home() / "huggingface"
TEXT_MODEL = HF_HOME / "Qwen3-0.6B"
VL_MODEL = HF_HOME / "Qwen3-VL-2B-Instruct"


def require_model(path: Path):
    """Skip decorator when a local model directory is missing."""
    return pytest.mark.skipif(not path.is_dir(), reason=f"missing model: {path}")
