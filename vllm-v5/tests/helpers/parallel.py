"""Parallel layout divisibility checks (shared by pytest + bench scripts)."""

from __future__ import annotations

import os
from pathlib import Path

from transformers import AutoConfig


def _text_cfg(model: str | os.PathLike[str]):
    c = AutoConfig.from_pretrained(str(model), trust_remote_code=True)
    return getattr(c, "text_config", c)


def tp_divisible(
    model: str | os.PathLike[str],
    tp: int,
) -> tuple[bool, str]:
    """Whether model dimensions allow this TP size (MoE uses intermediate sharding)."""
    t = _text_cfg(model)
    n_heads = int(t.num_attention_heads)
    n_kv = int(t.num_key_value_heads)
    inter = int(t.intermediate_size)
    if n_heads % tp != 0:
        return False, f"num_attention_heads={n_heads} not divisible by tp={tp}"
    if n_kv >= tp:
        if n_kv % tp != 0:
            return False, f"num_key_value_heads={n_kv} not divisible by tp={tp}"
    elif tp % n_kv != 0:
        return False, f"tp={tp} not divisible by num_key_value_heads={n_kv}"
    if inter % tp != 0:
        return False, f"intermediate_size={inter} not divisible by tp={tp}"

    moe_inter = getattr(t, "moe_intermediate_size", None)
    n_experts = int(getattr(t, "num_experts", 0) or 0)
    if moe_inter is not None and n_experts > 0:
        moe_inter = int(moe_inter)
        if moe_inter % tp != 0:
            return False, (
                f"moe_intermediate_size={moe_inter} not divisible by tp={tp}"
            )
        if tp > n_experts:
            return False, f"tp={tp} > num_experts={n_experts}"
        shared = int(getattr(t, "shared_expert_intermediate_size", 0) or 0)
        if shared and shared % tp != 0:
            return False, (
                f"shared_expert_intermediate_size={shared} not divisible by tp={tp}"
            )
    return True, "ok"


def pp_divisible(model: str | os.PathLike[str], pp: int) -> tuple[bool, str]:
    t = _text_cfg(model)
    n_layers = int(t.num_hidden_layers)
    if pp > n_layers:
        return False, f"pp={pp} > num_hidden_layers={n_layers}"
    if pp < 1:
        return False, "pp must be >= 1"
    return True, "ok"


def layout_ok(
    model: str | os.PathLike[str],
    *,
    tp: int = 1,
    pp: int = 1,
) -> tuple[bool, str]:
    ok, reason = tp_divisible(model, tp)
    if not ok:
        return ok, reason
    return pp_divisible(model, pp)
