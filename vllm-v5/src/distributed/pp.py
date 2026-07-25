from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import nn


def get_pp_indices(
    num_hidden_layers: int, pp_rank: int, pp_size: int
) -> tuple[int, int]:
    if pp_size < 1:
        raise ValueError(f"pp_size must be >= 1, got {pp_size}")
    if not (0 <= pp_rank < pp_size):
        raise ValueError(f"pp_rank must be in [0, {pp_size}), got {pp_rank}")
    if num_hidden_layers < 0:
        raise ValueError(f"num_hidden_layers must be >= 0, got {num_hidden_layers}")

    base = num_hidden_layers // pp_size
    rem = num_hidden_layers % pp_size
    if pp_rank < rem:
        start = pp_rank * (base + 1)
        end = start + base + 1
    else:
        start = rem * (base + 1) + (pp_rank - rem) * base
        end = start + base
    return start, end


@dataclass
class IntermediateTensors:
    hidden_states: torch.Tensor
    residual: torch.Tensor


class PPMissingLayer(nn.Identity):
    def forward(self, *args, **kwargs):
        return args[0] if args else next(iter(kwargs.values()))


def get_layer_id(weight_name: str) -> int | None:
    match = re.search(r"layers\.(\d+)\.", weight_name)
    if match:
        return int(match.group(1))
    return None
