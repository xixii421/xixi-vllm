from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight)


@contextmanager
def default_dtype_context(dtype: torch.dtype):
    """Construct modules under a target floating dtype (ullm / vLLM style)."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


STACKED_QKV = (
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
)
STACKED_GATE_UP = (
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
)


def move_module(
    module: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Move parameters and floating buffers to ``device``/``dtype``."""
    module.to(device=device, dtype=dtype)
    for name, buf in module.named_buffers():
        if buf is None or not buf.is_floating_point():
            continue
        parent, _, leaf = name.rpartition(".")
        mod = module.get_submodule(parent) if parent else module
        setattr(mod, leaf, buf.to(device=device, dtype=dtype))
    return module.eval()


def load_stacked_weight(
    *,
    name: str,
    loaded_weight: torch.Tensor,
    params_dict: dict[str, nn.Parameter],
    stacked_mapping: Iterable[tuple[str, str, Any]],
    loaded: set[str],
) -> bool:
    """Try stacked QKV / gate_up style mappings. Returns True if handled."""
    for param_name, weight_name, shard_id in stacked_mapping:
        if weight_name not in name:
            continue
        mapped = name.replace(weight_name, param_name)
        if mapped not in params_dict:
            break
        param = params_dict[mapped]
        getattr(param, "weight_loader")(param, loaded_weight, shard_id)
        loaded.add(mapped)
        return True
    return False


def load_plain_or_tied(
    *,
    name: str,
    loaded_weight: torch.Tensor,
    params_dict: dict[str, nn.Parameter],
    loaded: set[str],
    tie_word_embeddings: bool,
    embed_name: str = "model.embed_tokens.weight",
    lm_head_name: str = "lm_head.weight",
) -> bool:
    """Load a plain parameter, with optional tied-embedding fallback."""
    if name in params_dict:
        param = params_dict[name]
        getattr(param, "weight_loader", default_weight_loader)(param, loaded_weight)
        loaded.add(name)
        return True
    if (
        tie_word_embeddings
        and name == embed_name
        and lm_head_name in params_dict
        and embed_name not in params_dict
    ):
        param = params_dict[lm_head_name]
        getattr(param, "weight_loader", default_weight_loader)(param, loaded_weight)
        loaded.add(lm_head_name)
        return True
    return False


def finish_tied_embeddings(
    params_dict: dict[str, nn.Parameter],
    loaded: set[str],
    *,
    tie_word_embeddings: bool,
    embed_name: str = "model.embed_tokens.weight",
    lm_head_name: str = "lm_head.weight",
) -> None:
    if not tie_word_embeddings:
        return
    if lm_head_name not in params_dict or lm_head_name in loaded:
        return
    if embed_name not in params_dict:
        return
    lm = params_dict[lm_head_name]
    emb = params_dict[embed_name]
    if lm.data_ptr() == emb.data_ptr():
        loaded.add(lm_head_name)
        return
    default_weight_loader(lm, emb)
    loaded.add(lm_head_name)


def assert_weights_loaded(
    params_dict: dict[str, nn.Parameter],
    loaded: set[str],
    *,
    context: str,
) -> None:
    missing = [name for name in params_dict if name not in loaded]
    if not missing:
        return
    preview = ", ".join(missing[:8])
    suffix = " ..." if len(missing) > 8 else ""
    raise RuntimeError(
        f"{context}: missing {len(missing)} parameters after load: {preview}{suffix}"
    )
