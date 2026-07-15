"""vllm-v2 public API.

Layered layout (bottom → top)::

    layers/   → operators (Attention, RMSNorm, RoPE, …)
    models/   → Qwen3 graph + weight load
    memory/   → PagedAttention pool (BlockManager, PagedKVCache, slots)
    engine/   → generate loop + Sampler
    utils/    → ForwardContext bridging Engine ↔ Attention
    config.py → EngineConfig

Prefer explicit layer imports in library code::

    from src.memory import BlockManager
    from src.engine import Engine

``from src import Engine, EngineConfig`` remains supported via lazy ``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["Engine", "EngineConfig"]

if TYPE_CHECKING:
    from .config import EngineConfig as EngineConfig
    from .engine import Engine as Engine


def __getattr__(name: str) -> Any:
    if name == "EngineConfig":
        from .config import EngineConfig

        return EngineConfig
    if name == "Engine":
        from .engine import Engine

        return Engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
