from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int = 128
    ignore_eos: bool = False
