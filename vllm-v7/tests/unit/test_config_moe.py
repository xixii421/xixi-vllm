"""EngineConfig MoE TP / PP validation (uses tiny on-disk MoE checkpoint)."""

from __future__ import annotations

import pytest

from src.config import EngineConfig
from tests.helpers.moe_tiny import build_tiny_moe


@pytest.fixture(scope="module")
def tiny_moe_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_moe_cfg")
    return build_tiny_moe(d)


def test_moe_tp2_ok(tiny_moe_dir):
    cfg = EngineConfig(
        model=str(tiny_moe_dir),
        tp_size=2,
        pp_size=1,
        device_ids=[0, 1],
        enforce_eager=True,
        allow_hetero_tp=True,
    )
    assert cfg.is_moe and cfg.tp_size == 2


def test_moe_pp_too_large(tiny_moe_dir):
    with pytest.raises(ValueError, match="pp_size"):
        EngineConfig(
            model=str(tiny_moe_dir),
            tp_size=1,
            pp_size=8,
            device_ids=list(range(8)),
            allow_hetero_tp=True,
        )
