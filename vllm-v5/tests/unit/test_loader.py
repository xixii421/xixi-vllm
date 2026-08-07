"""Functional tests for safetensors_weights_iterator (v4-aligned fast path)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from src.models.loader import safetensors_weights_iterator
from src.utils.paths import resolve_hf_home


def _write_shard(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))


def _collect(folder: str) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in safetensors_weights_iterator(folder)}


def test_single_shard_roundtrip(tmp_path: Path):
    shard = {
        "layers.0.weight": torch.randn(8, 4),
        "layers.0.bias": torch.randn(8),
        "embed.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
    }
    _write_shard(tmp_path / "model.safetensors", shard)
    got = _collect(str(tmp_path))
    assert set(got) == set(shard)
    for k, v in shard.items():
        assert torch.equal(got[k], v)


def test_multi_shard_prefetch_order_and_values(tmp_path: Path):
    """Prefetch path must preserve sorted shard order and tensor equality."""
    a = {
        "a.weight": torch.ones(3, 3),
        "a.bias": torch.zeros(3),
    }
    b = {
        "b.weight": torch.full((2, 2), 2.0),
    }
    c = {
        "c.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
    }
    # Unsorted write names; iterator must sort.
    _write_shard(tmp_path / "model-00002-of-00003.safetensors", b)
    _write_shard(tmp_path / "model-00003-of-00003.safetensors", c)
    _write_shard(tmp_path / "model-00001-of-00003.safetensors", a)

    keys_in_order = [k for k, _ in safetensors_weights_iterator(str(tmp_path))]
    # Within a shard, safetensors may not preserve insertion order; shard order must.
    assert keys_in_order.index("a.weight") < keys_in_order.index("b.weight")
    assert keys_in_order.index("a.bias") < keys_in_order.index("b.weight")
    assert keys_in_order.index("b.weight") < keys_in_order.index("c.weight")
    assert set(keys_in_order) == {"a.weight", "a.bias", "b.weight", "c.weight"}

    got = _collect(str(tmp_path))
    expect = {**a, **b, **c}
    assert set(got) == set(expect)
    for k, v in expect.items():
        assert torch.equal(got[k], v)


def test_empty_folder_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        list(safetensors_weights_iterator(str(tmp_path)))


def test_device_kwarg_accepted(tmp_path: Path):
    _write_shard(tmp_path / "model.safetensors", {"w": torch.ones(2)})
    # API compat: device is ignored but must not break.
    got = dict(safetensors_weights_iterator(str(tmp_path), device="cuda"))
    assert "w" in got
    assert got["w"].device.type == "cpu"


@pytest.mark.skipif(
    not (resolve_hf_home() / "Qwen3-0.6B").is_dir(),
    reason="Qwen3-0.6B not cached locally",
)
def test_real_qwen06b_key_count_stable():
    folder = resolve_hf_home() / "Qwen3-0.6B"
    keys1 = [k for k, _ in safetensors_weights_iterator(folder)]
    keys2 = [k for k, _ in safetensors_weights_iterator(folder)]
    assert keys1 == keys2
    assert len(keys1) > 100
    # Spot-check a known HF name exists.
    assert any("embed_tokens" in k or "layers.0" in k for k in keys1)
