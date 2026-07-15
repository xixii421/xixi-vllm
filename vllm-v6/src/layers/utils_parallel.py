"""Pipeline-parallel helpers and Megatron-style math utils."""

from __future__ import annotations

import re

import torch


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


class IntermediateTensors(dict[str, torch.Tensor]):
    """Hold mid-pipeline activations (hidden_states, residual)."""

    pass


class PPMissingLayer(torch.nn.Identity):
    """Placeholder for layers not owned by this pipeline rank."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return args[0] if args else next(iter(kwargs.values()))


def get_layer_id(weight_name):
    # example: model.layers.10.self_attn.qkv_proj.weight
    match = re.search(r"layers\.(\d+)\.", weight_name)
    if match:
        return int(match.group(1))
    return None
