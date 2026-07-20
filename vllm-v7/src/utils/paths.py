
from __future__ import annotations

import os
from pathlib import Path

# FAST_HF_HOME = Path("/data/huggingface-local")
HUGGINGFACE_HUB_CACHE = Path("/mnt/nfs/zhangli/huggingface")


def resolve_hf_home() -> Path:

    for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return Path(raw).expanduser()
    return HUGGINGFACE_HUB_CACHE


def resolve_model_path(model: str | os.PathLike[str]) -> Path:
    path = Path(model).expanduser()
    if path.exists():
        return path
    return resolve_hf_home() / path.name
