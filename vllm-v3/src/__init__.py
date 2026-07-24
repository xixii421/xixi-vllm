"""vllm-v3 public API.

Layered layout (v2 three-plane + Fast data plane)::

    memory/  → KVManager (pages + pool + hash) + Sequence + Triton KV
    engine/  → Scheduler + ModelRunner (FlashInfer/CG) + Engine
    layers/  → FlashInfer Attention + Triton Norm/RoPE
    core/    → CUDA Graph decode runner
    models/  → Qwen3
    config.py

``from src import Engine, EngineConfig, SamplingParams`` via lazy ``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["Engine", "EngineConfig", "SamplingParams"]

if TYPE_CHECKING:
    from .config import EngineConfig as EngineConfig
    from .engine import Engine as Engine
    from .sampling_params import SamplingParams as SamplingParams


def __getattr__(name: str) -> Any:
    if name == "EngineConfig":
        from .config import EngineConfig

        return EngineConfig
    if name == "Engine":
        from .engine import Engine

        return Engine
    if name == "SamplingParams":
        from .sampling_params import SamplingParams

        return SamplingParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
