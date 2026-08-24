"""Distributed plane: process groups, collectives, parallel helpers.

仅暴露 TP/PP 相关 API。DP 是控制面副本（``MPClient``），不在此建 NCCL 组。
PP 点对点请用阻塞 ``send_tensors`` / ``recv_tensors``（piecewise CG 图外调用）。
"""

from __future__ import annotations

from .communication_op import (
    all_gather,
    all_reduce,
    recv_tensors,
    reduce_scatter,
    send_tensors,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from .control_channel import ControlChannel
from .parallel_state import (
    get_pp_group,
    get_tp_group,
    get_world_group,
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
    is_initialized,
    destroy_distributed_environment,
    destroy_model_parallel,
    set_nccl_env,
)
from .utils import (
    IntermediateTensors,
    PPMissingLayer,
    divide,
    ensure_divisibility,
    get_pp_indices,
    set_pp_layer_ranges,
)

__all__ = [
    "all_gather",
    "all_reduce",
    "recv_tensors",
    "reduce_scatter",
    "send_tensors",
    "tensor_model_parallel_all_gather",
    "tensor_model_parallel_all_reduce",
    "tensor_model_parallel_reduce_scatter",
    "get_pp_group",
    "get_tp_group",
    "get_world_group",
    "ControlChannel",
    "graph_capture",
    "init_distributed_environment",
    "initialize_model_parallel",
    "is_initialized",
    "destroy_distributed_environment",
    "destroy_model_parallel",
    "set_nccl_env",
    "IntermediateTensors",
    "PPMissingLayer",
    "divide",
    "ensure_divisibility",
    "get_pp_indices",
    "set_pp_layer_ranges",
]
