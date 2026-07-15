from __future__ import annotations
import re
import torch

def get_pp_indices(num_hidden_layers: int, pp_rank: int, pp_size: int) -> tuple[int, int]:
    layers_per_partition = num_hidden_layers // pp_size
    start_layer = pp_rank * layers_per_partition
    end_layer = start_layer + layers_per_partition
    if pp_rank == pp_size - 1:
        end_layer = num_hidden_layers
    return (start_layer, end_layer)

class IntermediateTensors(dict[str, torch.Tensor]):
    pass

class PPMissingLayer(torch.nn.Identity):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return args[0] if args else next(iter(kwargs.values()))

def get_layer_id(weight_name):
    match = re.search('layers\\.(\\d+)\\.', weight_name)
    if match:
        return int(match.group(1))
    return None
