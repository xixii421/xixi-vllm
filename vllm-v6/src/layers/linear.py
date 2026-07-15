import torch
import torch.nn as nn
import torch.nn.functional as F
from ..distributed import get_tp_group, tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce
from ..distributed.math import divide, ensure_divisibility

class ReplicatedLinear(nn.Module):

    def __init__(self, input_size: int, output_size: int, bias: bool=True, skip_bias_add: bool=False, params_dtype: torch.dtype | None=None, return_bias: bool=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype
        self.return_bias = return_bias
        tp_size = get_tp_group().size
        tp_rank = get_tp_group().group_rank
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size, dtype=params_dtype))
        setattr(self.weight, 'weight_loader', self.weight_loader)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size, dtype=self.params_dtype))
            setattr(self.bias, 'weight_loader', self.weight_loader)
        else:
            self.register_parameter('bias', None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        assert param.shape == loaded_weight.shape, f'param.shape={param.shape!r} != loaded_weight.shape={loaded_weight.shape!r}'
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        bias = self.bias if not self.skip_bias_add else None
        output = F.linear(x, self.weight, bias)
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return (output, output_bias)

class ColumnParallelLinear(nn.Module):

    def __init__(self, input_size: int, output_size: int, bias: bool=True, gather_output: bool=True, skip_bias_add: bool=False, params_dtype: torch.dtype | None=None, return_bias: bool=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype
        self.return_bias = return_bias
        tp_size = get_tp_group().size
        tp_rank = get_tp_group().group_rank
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.output_size_per_partition = divide(output_size, tp_size)
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size, dtype=params_dtype))
        setattr(self.weight, 'weight_loader', self.weight_loader)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition, dtype=params_dtype))
            setattr(self.bias, 'weight_loader', self.weight_loader)
        else:
            self.register_parameter('bias', None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        output_dim = 0
        param_data = param.data
        shard_size = param_data.shape[output_dim]
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape, f'param_data.shape={param_data.shape!r} != loaded_weight.shape={loaded_weight.shape!r}'
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(x, self.weight, bias)
        if self.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return (output, output_bias)

class RowParallelLinear(nn.Module):

    def __init__(self, input_size: int, output_size: int, bias: bool=True, input_is_parallel: bool=True, skip_bias_add: bool=False, params_dtype: torch.dtype | None=None, reduce_results: bool=True, return_bias: bool=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype
        self.reduce_results = reduce_results
        self.return_bias = return_bias
        tp_size = get_tp_group().size
        tp_rank = get_tp_group().group_rank
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.input_size_per_partition = divide(input_size, tp_size)
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition, dtype=params_dtype))
        setattr(self.weight, 'weight_loader', self.weight_loader)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size, dtype=params_dtype))
            setattr(self.bias, 'weight_loader', self.weight_loader)
        else:
            self.register_parameter('bias', None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        input_dim = 1
        param_data = param.data
        shard_size = param_data.shape[input_dim]
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape, f'param_data.shape={param_data.shape!r} != loaded_weight.shape={loaded_weight.shape!r}'
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.input_is_parallel:
            input_parallel = x
        else:
            splitted_input = torch.split(x, self.input_size_per_partition, dim=-1)
            input_parallel = splitted_input[self.tp_rank].contiguous()
        bias = None if self.tp_rank > 0 or self.skip_bias_add else self.bias
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return (output, output_bias)

class QKVParallelLinear(ColumnParallelLinear):

    def __init__(self, hidden_size: int, head_size: int, total_num_heads: int, total_num_kv_heads: int | None=None, bias: bool=True, skip_bias_add: bool=False, params_dtype: torch.dtype | None=None, return_bias: bool=True):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads if total_num_kv_heads is not None else total_num_heads
        self.params_dtype = params_dtype
        tp_size = get_tp_group().size
        self.num_heads = divide(total_num_heads, tp_size)
        if tp_size > self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = hidden_size
        output_size = (self.num_heads + 2 * self.num_kv_heads) * tp_size * self.head_size
        super().__init__(input_size=input_size, output_size=output_size, bias=bias, gather_output=False, skip_bias_add=skip_bias_add, params_dtype=params_dtype, return_bias=return_bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str | None=None):
        output_dim = 0
        param_data = param.data
        if loaded_shard_id is None:
            shard_offsets = [('q', 0, self.total_num_heads * self.head_size), ('k', self.total_num_heads * self.head_size, self.total_num_kv_heads * self.head_size), ('v', (self.total_num_heads + self.total_num_kv_heads) * self.head_size, self.total_num_kv_heads * self.head_size)]
            for shard_id, shard_offset, shard_size in shard_offsets:
                loaded_weight_shard = loaded_weight.narrow(output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return
        if loaded_shard_id == 'q':
            shard_offset = 0
            shard_size = self.num_heads * self.head_size
        elif loaded_shard_id == 'k':
            shard_offset = self.num_heads * self.head_size
            shard_size = self.num_kv_heads * self.head_size
        elif loaded_shard_id == 'v':
            shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
            shard_size = self.num_kv_heads * self.head_size
        else:
            assert False, f'Invalid shard_id {loaded_shard_id}'
        param_data = param_data.narrow(output_dim, shard_offset, shard_size)
        if loaded_shard_id == 'q':
            shard_id = self.tp_rank
        else:
            shard_id = self.tp_rank // self.num_kv_head_replicas
        start_idx = shard_id * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape, f'param_data.shape={param_data.shape!r} != loaded_weight.shape={loaded_weight.shape!r}'
        param_data.copy_(loaded_weight)

class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool=True, gather_output: bool=False, skip_bias_add: bool=False, params_dtype: torch.dtype | None=None):
        self.output_sizes = output_sizes
        tp_size = get_tp_group().size
        for output_size in output_sizes:
            ensure_divisibility(output_size, tp_size)
        super().__init__(input_size=input_size, output_size=sum(output_sizes), bias=bias, gather_output=gather_output, skip_bias_add=skip_bias_add, params_dtype=params_dtype)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int | None=None):
        output_dim = 0
        param_data = param.data
        if loaded_shard_id is None:
            current_shard_offset = 0
            shard_offsets: list[tuple[int, int, int]] = []
            for i, output_size in enumerate(self.output_sizes):
                shard_offsets.append((i, current_shard_offset, output_size))
                current_shard_offset += output_size
            for shard_id, shard_offset, shard_size in shard_offsets:
                loaded_weight_shard = loaded_weight.narrow(output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return
        assert loaded_shard_id < len(self.output_sizes), f'Invalid shard_id {loaded_shard_id}'
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(output_dim, shard_offset, shard_size)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape, f'param_data.shape={param_data.shape!r} != loaded_weight.shape={loaded_weight.shape!r}'
        param_data.copy_(loaded_weight)
