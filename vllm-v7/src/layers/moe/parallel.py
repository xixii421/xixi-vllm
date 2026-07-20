"""TP shard + fused backend。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MoeBackend = Literal["cutlass", "triton"]


@dataclass(frozen=True)
class MoEParallelConfig:
    tp_size: int
    tp_rank: int
    fused_backend: MoeBackend = "triton"

    @classmethod
    def make(
        cls,
        *,
        tp_size: int,
        tp_rank: int,
        fused_backend: MoeBackend = "triton",
    ) -> "MoEParallelConfig":
        if tp_size < 1:
            raise ValueError("tp_size must be >= 1")
        if not 0 <= tp_rank < tp_size:
            raise ValueError(f"tp_rank={tp_rank} out of range for tp_size={tp_size}")
        if fused_backend not in ("cutlass", "triton"):
            raise ValueError(f"fused_backend must be cutlass|triton, got {fused_backend!r}")
        return cls(tp_size=tp_size, tp_rank=tp_rank, fused_backend=fused_backend)


_MOE_PARALLEL: MoEParallelConfig | None = None


def set_moe_parallel_config(cfg: MoEParallelConfig | None) -> None:
    global _MOE_PARALLEL
    _MOE_PARALLEL = cfg


def get_moe_parallel_config() -> MoEParallelConfig:
    if _MOE_PARALLEL is not None:
        return _MOE_PARALLEL
    return MoEParallelConfig.make(tp_size=1, tp_rank=0, fused_backend="triton")
