"""Qwen3 权重缓存路径辅助。"""

from __future__ import annotations

import os
from pathlib import Path

# 缓存根目录：优先环境变量，其次退回 ~/huggingface
DEFAULT_CACHE_DIR = Path(
    os.environ.get("HF_HOME")
    or os.environ.get("HUGGINGFACE_HUB_CACHE")
    or Path.home() / "huggingface"
)
