"""Distributed 通用工具：整除、层划分、跨 stage 中间张量与占位层。"""

from __future__ import annotations

import torch

# 不均分 PP 时各 rank 的 (start_layer, end_layer)。由 worker 在 init_environment
# 里按 config.pp_layer_counts 写入；None 时回退到等分。放在模块级而非 parallel_state，
# 避免循环 import，且 get_pp_indices 是本模块纯函数、便于单测。
_PP_LAYER_RANGES: list[tuple[int, int]] | None = None


def ensure_divisibility(numerator: int, denominator: int) -> None:
    assert numerator % denominator == 0, (
        f"{numerator} is not divisible by {denominator}"
    )


def divide(numerator: int, denominator: int) -> int:
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def _equal_split(
    num_hidden_layers: int, pp_rank: int, pp_size: int
) -> tuple[int, int]:
    """均分隐藏层；不能整除时最后一档吃掉余数。"""
    layers_per_partition = num_hidden_layers // pp_size
    start_layer = pp_rank * layers_per_partition
    end_layer = start_layer + layers_per_partition
    if pp_rank == pp_size - 1:
        end_layer = num_hidden_layers
    return start_layer, end_layer


def get_pp_indices(
    num_hidden_layers: int, pp_rank: int, pp_size: int
) -> tuple[int, int]:
    """返回本 rank 的层区间；若已设置不均分区间则优先使用。"""
    if _PP_LAYER_RANGES is not None:
        return _PP_LAYER_RANGES[pp_rank]
    return _equal_split(num_hidden_layers, pp_rank, pp_size)


def set_pp_layer_ranges(layer_counts: list[int] | None) -> None:
    """按每 stage 层数列表设置不均分区间（累加得到 start/end）。None 复位为等分。"""
    global _PP_LAYER_RANGES
    if layer_counts is None:
        _PP_LAYER_RANGES = None
        return
    ranges: list[tuple[int, int]] = []
    start = 0
    for c in layer_counts:
        ranges.append((start, start + c))
        start += c
    _PP_LAYER_RANGES = ranges


class IntermediateTensors(dict[str, torch.Tensor]):
    """PP 相邻 stage 之间传递的中间张量（按名字索引）。"""

    pass


class PPMissingLayer(torch.nn.Identity):
    """本 rank 不持有的层的占位符，forward 原样透传第一个参数。"""

    def forward(self, *args, **kwargs):  # noqa: D102
        return args[0] if args else next(iter(kwargs.values()))
