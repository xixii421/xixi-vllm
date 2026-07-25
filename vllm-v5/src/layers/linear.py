import torch
import torch.nn as nn
import torch.nn.functional as F

from ..distributed import (
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from ..distributed.math import divide, ensure_divisibility


def _copy_weight(dest: torch.Tensor, source: torch.Tensor) -> None:
    assert dest.shape == source.shape, f"{dest.shape} != {source.shape}"
    dest.copy_(source)


class ReplicatedLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=params_dtype)
        )
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size, dtype=params_dtype))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        _copy_weight(param.data, loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = True,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.gather_output = gather_output
        tp = get_tp_group()
        self.tp_size = tp.size
        self.tp_rank = tp.group_rank
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, input_size, dtype=params_dtype)
        )
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.output_size_per_partition, dtype=params_dtype)
            )
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        param_data = param.data
        shard_size = param_data.shape[0]
        shard = loaded_weight.narrow(0, self.tp_rank * shard_size, shard_size)
        _copy_weight(param_data, shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_parallel = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            return tensor_model_parallel_all_gather(output_parallel)
        return output_parallel


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.reduce_results = reduce_results
        tp = get_tp_group()
        self.tp_size = tp.size
        self.tp_rank = tp.group_rank
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.weight = nn.Parameter(
            torch.empty(output_size, self.input_size_per_partition, dtype=params_dtype)
        )
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size, dtype=params_dtype))
            # Row-parallel bias is replicated, not sharded with the weight.
            self.bias.weight_loader = self.bias_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        param_data = param.data
        shard_size = param_data.shape[1]
        shard = loaded_weight.narrow(1, self.tp_rank * shard_size, shard_size)
        _copy_weight(param_data, shard)

    def bias_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        _copy_weight(param.data, loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bias is added only on rank 0 so it is not duplicated by all-reduce.
        bias = None if self.tp_rank > 0 else self.bias
        output_parallel = F.linear(x, self.weight, bias)
        if self.reduce_results and self.tp_size > 1:
            return tensor_model_parallel_all_reduce(output_parallel)
        return output_parallel


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = (
            total_num_kv_heads
            if total_num_kv_heads is not None
            else total_num_heads
        )
        tp_size = get_tp_group().size
        self.num_heads = divide(total_num_heads, tp_size)
        if tp_size > self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        output_size = (
            (self.num_heads + 2 * self.num_kv_heads) * tp_size * head_size
        )
        super().__init__(
            input_size=hidden_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            params_dtype=params_dtype,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        output_dim = 0
        param_data = param.data
        if loaded_shard_id is None:
            q_size = self.total_num_heads * self.head_size
            kv_size = self.total_num_kv_heads * self.head_size
            shard_offsets = (
                ("q", 0, q_size),
                ("k", q_size, kv_size),
                ("v", q_size + kv_size, kv_size),
            )
            for shard_id, shard_offset, shard_size in shard_offsets:
                shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size
                )
                self.weight_loader(param, shard, shard_id)
            return
        if loaded_shard_id == "q":
            shard_offset = 0
            shard_size = self.num_heads * self.head_size
        elif loaded_shard_id == "k":
            shard_offset = self.num_heads * self.head_size
            shard_size = self.num_kv_heads * self.head_size
        elif loaded_shard_id == "v":
            shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
            shard_size = self.num_kv_heads * self.head_size
        else:
            raise ValueError(f"invalid shard id: {loaded_shard_id}")
        param_data = param_data.narrow(output_dim, shard_offset, shard_size)
        if loaded_shard_id == "q":
            shard_id = self.tp_rank
        else:
            shard_id = self.tp_rank // self.num_kv_head_replicas
        start_idx = shard_id * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        _copy_weight(param_data, loaded_weight)


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        self.output_sizes = output_sizes
        tp_size = get_tp_group().size
        for output_size in output_sizes:
            ensure_divisibility(output_size, tp_size)
        super().__init__(
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
            gather_output=gather_output,
            params_dtype=params_dtype,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | None = None,
    ) -> None:
        output_dim = 0
        param_data = param.data
        if loaded_shard_id is None:
            current_shard_offset = 0
            shard_offsets: list[tuple[int, int, int]] = []
            for i, output_size in enumerate(self.output_sizes):
                shard_offsets.append((i, current_shard_offset, output_size))
                current_shard_offset += output_size
            for shard_id, shard_offset, shard_size in shard_offsets:
                shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size
                )
                self.weight_loader(param, shard, shard_id)
            return
        assert loaded_shard_id < len(self.output_sizes)
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(output_dim, shard_offset, shard_size)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        _copy_weight(param_data, loaded_weight)
