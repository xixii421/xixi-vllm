from __future__ import annotations

import torch
from torch import nn

from ...distributed.math import divide, ensure_divisibility
from .fused import fused_moe_forward
from .parallel import get_moe_parallel_config
from .weights import hf_gate_up_to_cutlass, normalize_fused_weight, w13_slice


class Experts(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        tp_size: int,
        tp_rank: int,
    ) -> None:
        super().__init__()
        ensure_divisibility(intermediate_size, tp_size)
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.intermediate_size_per_partition = divide(intermediate_size, tp_size)
        i = self.intermediate_size_per_partition
        # 存 cutlass 布局 [up; gate]
        self.w13_weight = nn.Parameter(torch.empty(num_experts, 2 * i, hidden_size))
        self.w2_weight = nn.Parameter(torch.empty(num_experts, hidden_size, i))
        setattr(self.w13_weight, "weight_loader", self.weight_loader_w13)
        setattr(self.w2_weight, "weight_loader", self.weight_loader_w2)

    def weight_loader_w13(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        expert_id: int,
    ) -> None:
        shard = self.intermediate_size_per_partition
        start = self.tp_rank * shard
        piece = loaded_weight.narrow(0, start, shard)
        dest = param.data[expert_id, w13_slice(weight_name, shard)]
        assert dest.shape == piece.shape, f"{dest.shape} != {piece.shape}"
        dest.copy_(piece)

    def weight_loader_w2(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        expert_id: int,
    ) -> None:
        shard = self.intermediate_size_per_partition
        start = self.tp_rank * shard
        piece = loaded_weight.narrow(1, start, shard)
        dest = param.data[expert_id]
        assert dest.shape == piece.shape, f"{dest.shape} != {piece.shape}"
        dest.copy_(piece)

    def load_fused_w13(self, loaded: torch.Tensor) -> None:
        loaded = normalize_fused_weight(
            loaded,
            (self.num_experts, 2 * self.intermediate_size, self.hidden_size),
            "gate_up",
        )
        loaded = hf_gate_up_to_cutlass(loaded)
        shard = self.intermediate_size_per_partition
        start = self.tp_rank * shard
        u = loaded[:, start : start + shard, :]
        g = loaded[
            :,
            self.intermediate_size + start : self.intermediate_size + start + shard,
            :,
        ]
        self.w13_weight.data.copy_(torch.cat([u, g], dim=1))

    def load_fused_w2(self, loaded: torch.Tensor) -> None:
        loaded = normalize_fused_weight(
            loaded,
            (self.num_experts, self.hidden_size, self.intermediate_size),
            "down",
        )
        shard = self.intermediate_size_per_partition
        start = self.tp_rank * shard
        self.w2_weight.data.copy_(loaded[:, :, start : start + shard])

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if not hidden_states.is_cuda:
            raise RuntimeError("MoE fused kernels require CUDA tensors")
        backend = get_moe_parallel_config().fused_backend
        return fused_moe_forward(
            backend=backend,
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13_weight=self.w13_weight,
            w2_weight=self.w2_weight,
        )
