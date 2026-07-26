"""Model layer: Qwen3 definition + weight loading.

Download / verify helpers live in ``src.models.hub`` (imported lazily there
to keep ``engine → models`` acyclic).
"""

from __future__ import annotations

from .loader import default_weight_loader, load_model
from .qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3Model,
)

__all__ = [
    "Qwen3ForCausalLM",
    "Qwen3Model",
    "Qwen3DecoderLayer",
    "Qwen3Attention",
    "Qwen3MLP",
    "load_model",
    "default_weight_loader",
]
