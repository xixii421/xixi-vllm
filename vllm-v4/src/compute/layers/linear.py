"""TP 分片线性层。

每种层都实现 ``weight_loader``，在权重加载时按 tp_rank 切分全量权重。
- ColumnParallel：输出维切分，可选 all-gather。
- RowParallel：输入维切分，输出前 all-reduce。
- QKVParallel：q/k/v 三阵按 head 切分（KV head 少于 tp 时复制）。
- MergedColumnParallel：gate/up 融合阵按输出维切分。
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from ...distributed import (
    divide,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)


def _shard(weight: torch.Tensor, dim: int) -> torch.Tensor:
    tp = get_tp_group()
    size = weight.shape[dim] // tp.size
    offset = size * tp.group_rank
    return weight.narrow(dim, offset, size).contiguous()


class ColumnParallelLinear(nn.Module):
    """输出维切分：weight [out_size // tp, in_size]。gather_output 时拼接完整输出。"""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        bias: bool = False,
        gather_output: bool = True,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.gather_output = gather_output
        self.output_size_per_partition = divide(out_size, get_tp_group().size)
        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, in_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
        else:
            self.register_parameter("bias", None)
        setattr(self.weight, "weight_loader", self.weight_loader)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int = 0
    ) -> None:
        param.data.copy_(_shard(loaded_weight, dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.gather_output:
            return tensor_model_parallel_all_gather(out, dim=-1)
        return out


class RowParallelLinear(nn.Module):
    """输入维切分：weight [out_size, in_size // tp]。输出前 all-reduce 求和。

    ``input_is_parallel=True`` 表示输入已按 head 切分（如 attention 的 o_proj，
    每个 rank 只持自己 head 的输出），此时不再二次切分输入维。
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        bias: bool = False,
        input_is_parallel: bool = False,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.input_is_parallel = input_is_parallel
        self.input_size_per_partition = (
            in_size if input_is_parallel else divide(in_size, get_tp_group().size)
        )
        self.weight = nn.Parameter(
            torch.empty(out_size, self.input_size_per_partition)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_size))
        else:
            self.register_parameter("bias", None)
        setattr(self.weight, "weight_loader", self.weight_loader)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int = 0
    ) -> None:
        param.data.copy_(_shard(loaded_weight, dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.linear(x, self.weight)
        out = tensor_model_parallel_all_reduce(out)
        if self.bias is not None:
            out = out + self.bias
        return out


class QKVParallelLinear(nn.Module):
    """q/k/v 融合并分片：weight [(n_q + 2 n_kv) * head_size, hidden]。

    输出不 gather，forward 返回各 rank 自己的 qkv，再在模型中 split。
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        *,
        bias: bool = False,
    ):
        super().__init__()
        tp_size = get_tp_group().size
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        if tp_size > total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, total_num_kv_heads)
        else:
            self.num_kv_heads = divide(total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        self.q_size = self.num_heads * head_size
        self.kv_size = self.num_kv_heads * head_size
        # 每个 rank 只持有自己的 q/k/v 段。
        self.weight = nn.Parameter(
            torch.empty(self.q_size + 2 * self.kv_size, hidden_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.q_size + 2 * self.kv_size))
        else:
            self.register_parameter("bias", None)
        setattr(self.weight, "weight_loader", self.weight_loader)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str
    ) -> None:
        tp_rank = get_tp_group().group_rank
        assert shard_id in ("q", "k", "v")
        if shard_id == "q":
            begin = 0
            shard = loaded_weight[
                tp_rank * self.q_size : (tp_rank + 1) * self.q_size
            ]
        else:
            if shard_id == "k":
                begin = self.q_size
            else:
                begin = self.q_size + self.kv_size
            if self.num_kv_head_replicas > 1:
                # KV head 少于 tp：同一条 head 被连续 num_kv_head_replicas 个 rank 复制。
                # 例：nkv=2/tp=4 时 head0 归 rank0,1、head1 归 rank2,3，故用整除分组。
                head_idx = tp_rank // self.num_kv_head_replicas
                start = head_idx * self.kv_size
            else:
                start = tp_rank * self.kv_size
            shard = loaded_weight[start : start + self.kv_size]
        param.data[begin : begin + shard.shape[0]].copy_(shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(nn.Module):
    """多个列并行阵融合为一个大阵（如 gate+up），按输出维切分。

    ``output_sizes`` 为每个子阵的全量输出维；rank 持有每个子阵的 1/tp。
    """

    def __init__(
        self, in_size: int, output_sizes: list[int], *, bias: bool = False
    ):
        super().__init__()
        self.in_size = in_size
        self.output_sizes = output_sizes
        self.output_sizes_per_partition = [
            divide(s, get_tp_group().size) for s in output_sizes
        ]
        total = sum(self.output_sizes_per_partition)
        self.weight = nn.Parameter(torch.empty(total, in_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(total))
        else:
            self.register_parameter("bias", None)
        setattr(self.weight, "weight_loader", self.weight_loader)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int
    ) -> None:
        shard_offset = sum(self.output_sizes_per_partition[:shard_id])
        shard_size = self.output_sizes_per_partition[shard_id]
        param.data[shard_offset : shard_offset + shard_size].copy_(
            _shard(loaded_weight, dim=0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


def _default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight)


def get_weight_loader(param: nn.Parameter) -> Callable:
    return getattr(param, "weight_loader", _default_weight_loader)
