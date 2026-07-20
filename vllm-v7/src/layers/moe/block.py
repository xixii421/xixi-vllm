from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...distributed import get_tp_group, tensor_model_parallel_all_reduce
from ..activation import SiluAndMul
from ..linear import MergedColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from .experts import Experts
from .parallel import MoEParallelConfig, get_moe_parallel_config
from .routing import route_experts


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config, parallel: MoEParallelConfig | None = None) -> None:
        super().__init__()
        par = parallel or get_moe_parallel_config()
        self.parallel = par
        hidden = config.hidden_size
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", False))

        self.gate = ReplicatedLinear(hidden, self.num_experts, bias=False)
        self.experts = Experts(
            self.num_experts,
            hidden,
            int(config.moe_intermediate_size),
            tp_size=par.tp_size,
            tp_rank=par.tp_rank,
        )
        self._init_shared_expert(config, hidden)

    def _init_shared_expert(self, config, hidden: int) -> None:
        shared_inter = int(getattr(config, "shared_expert_intermediate_size", 0) or 0)
        if shared_inter <= 0:
            self.shared_expert_gate = None
            self.shared_expert_gate_up = None
            self.shared_expert_down = None
            self.shared_activation = None
            return
        self.shared_expert_gate = ReplicatedLinear(hidden, 1, bias=False)
        self.shared_expert_gate_up = MergedColumnParallelLinear(
            hidden, [shared_inter] * 2, bias=False
        )
        self.shared_expert_down = RowParallelLinear(
            shared_inter, hidden, bias=False, reduce_results=False
        )
        self.shared_activation = SiluAndMul()

    def _shared_expert(self, hidden: torch.Tensor) -> torch.Tensor:
        assert self.shared_expert_gate_up is not None
        assert self.shared_expert_down is not None
        assert self.shared_activation is not None
        assert self.shared_expert_gate is not None
        shared = self.shared_expert_down(
            self.shared_activation(self.shared_expert_gate_up(hidden))
        )
        return F.sigmoid(self.shared_expert_gate(hidden)) * shared

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden = hidden_states.view(-1, orig_shape[-1])
        router_logits = self.gate(hidden)
        topk_weights, topk_ids = route_experts(
            router_logits, self.top_k, norm_topk_prob=self.norm_topk_prob
        )
        out = self.experts(hidden, topk_ids, topk_weights)
        if self.shared_expert_gate_up is not None:
            out = out + self._shared_expert(hidden)
        if get_tp_group().size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out.view(orig_shape)
