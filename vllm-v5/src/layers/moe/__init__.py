"""MoE layers: cutlass / Triton fused experts + TP。

Layout::

    block.py       Sparse MoE block
    routing.py     Softmax top-k
    experts.py     TP-sharded experts
    fused.py       cutlass / triton 分发
    align.py       moe_align（上界 alloc，对齐 vLLM）
    triton_moe.py  Triton GEMM + silu_and_mul + sum
    weights.py     CUTLASS [up; gate] 布局
    parallel.py    tp + fused_backend
"""

from __future__ import annotations

from .block import Qwen3MoeSparseMoeBlock
from .experts import Experts
from .fused import clear_fused_state
from .parallel import MoEParallelConfig, get_moe_parallel_config, set_moe_parallel_config
from .routing import route_experts

__all__ = [
    "Qwen3MoeSparseMoeBlock",
    "Experts",
    "route_experts",
    "MoEParallelConfig",
    "get_moe_parallel_config",
    "set_moe_parallel_config",
    "clear_fused_state",
]
