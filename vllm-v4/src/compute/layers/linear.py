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
    """返回本 rank 的权重分片 view（零拷贝）。

    不做 .contiguous()：dim=0 切分时 view 每行内部连续，copy_ 即逐行 memcpy；
    dim=1 切分时直接对 strided view 做 copy_ 也比「先 contiguous 再 copy」
    快约 1.7x（实测），因为省去一次完整临时缓冲区的分配与拷贝。
    """
    tp = get_tp_group()
    size = weight.shape[dim] // tp.size
    offset = size * tp.group_rank
    return weight.narrow(dim, offset, size)


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
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)
        setattr(self.weight, "weight_loader", self.weight_loader)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int = 0
    ) -> None:
        # bias 与 weight 同为输出维切分（全量 [out] -> 本地 [out//tp]）。
        param.data.copy_(_shard(loaded_weight, dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.gather_output:
            return tensor_model_parallel_all_gather(out, dim=-1)
        return out


class RowParallelLinear(nn.Module):
    """输入维切分：weight [out_size, in_size // tp]。输出前 all-reduce 求和。

    input_is_parallel=True 表示输入已按 head 切分（如 attention 的 o_proj，
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
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)
        setattr(self.weight, "weight_loader", self.weight_loader)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int = 0
    ) -> None:
        if param is self.bias:
            # 行并行输出维不切分，bias 全量持有。
            param.data.copy_(loaded_weight)
            return
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
        self.num_heads_total = total_num_heads
        self.num_kv_heads_total = total_num_kv_heads
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
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)
        setattr(self.weight, "weight_loader", self.weight_loader)

    def _shard_qkv(
        self, loaded_weight: torch.Tensor, shard_id: str
    ) -> tuple[int, torch.Tensor]:
        """按 shard_id 切出本 rank 的 q/k/v 段，返回 (本地起始行, 段)。

        loaded_weight 是单个 q/k/v 的全量（HF 的 q_proj/k_proj/v_proj），
        begin 是写入本地 [q_size + 2*kv_size] 参数的偏移。
        """
        tp_rank = get_tp_group().group_rank
        if shard_id == "q":
            begin = 0
            return begin, loaded_weight[
                tp_rank * self.q_size : (tp_rank + 1) * self.q_size
            ]
        if shard_id == "k":
            begin = self.q_size
        else:  # "v"
            begin = self.q_size + self.kv_size
        if self.num_kv_head_replicas > 1:
            # KV head 少于 tp：同一条 head 被连续 num_kv_head_replicas 个 rank 复制。
            # 例：nkv=2/tp=4 时 head0 归 rank0,1、head1 归 rank2,3，故用整除分组。
            head_idx = tp_rank // self.num_kv_head_replicas
            start = head_idx * self.kv_size
        else:
            start = tp_rank * self.kv_size
        return begin, loaded_weight[start : start + self.kv_size]

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str
    ) -> None:
        assert shard_id in ("q", "k", "v", "bias")
        if shard_id == "bias":
            self._load_bias(loaded_weight)
            return
        begin, shard = self._shard_qkv(loaded_weight, shard_id)
        param.data[begin : begin + shard.shape[0]].copy_(shard)

    def _load_bias(self, loaded_weight: torch.Tensor) -> None:
        """加载合并形式的 qkv bias（[q 全量 | k 全量 | v 全量] 单段）。"""
        q_total = self.num_heads_total * self.head_size
        kv_total = self.num_kv_heads_total * self.head_size
        _, q_shard = self._shard_qkv(loaded_weight[:q_total], "q")
        self.bias.data[: self.q_size].copy_(q_shard)
        k_seg = loaded_weight[q_total : q_total + kv_total]
        v_seg = loaded_weight[q_total + kv_total : q_total + 2 * kv_total]
        for idx, (seg, shard_id) in enumerate(((k_seg, "k"), (v_seg, "v"))):
            _, shard = self._shard_qkv(seg, shard_id)
            lo = self.q_size + idx * self.kv_size
            self.bias.data[lo : lo + self.kv_size].copy_(shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(nn.Module):
    """多个列并行阵融合为一个大阵（如 gate+up），按输出维切分。

    output_sizes 为每个子阵的全量输出维；rank 持有每个子阵的 1/tp。
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
            setattr(self.bias, "weight_loader", self.weight_loader)
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
