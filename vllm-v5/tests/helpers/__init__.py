"""Shared test / bench utilities."""

from __future__ import annotations

from .devices import blackwell_a100_pair, find_a100, find_blackwell
from .engine import (
    cfg,
    decode_token_ids,
    engine_session,
    free_cuda,
    greedy,
    greedy_tokens,
    greedy_tokens_with_engine,
    greedy_with_engine,
    homo_device_ids,
    measure,
    measure_with_engine,
    solid_image,
    tp_divisible,
)
from .parallel import layout_ok, pp_divisible
from .paths import HF_HOME, MOE_MODEL, TEXT_MODEL, VL_MODEL, require_model

__all__ = [
    "HF_HOME",
    "TEXT_MODEL",
    "VL_MODEL",
    "MOE_MODEL",
    "require_model",
    "cfg",
    "decode_token_ids",
    "engine_session",
    "free_cuda",
    "greedy",
    "greedy_tokens",
    "greedy_tokens_with_engine",
    "greedy_with_engine",
    "homo_device_ids",
    "measure",
    "measure_with_engine",
    "solid_image",
    "tp_divisible",
    "pp_divisible",
    "layout_ok",
    "blackwell_a100_pair",
    "find_a100",
    "find_blackwell",
]
