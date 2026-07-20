from __future__ import annotations
import torch

def hf_gate_up_to_cutlass(w13_hf: torch.Tensor) -> torch.Tensor:
    gate, up = w13_hf.chunk(2, dim=1)
    return torch.cat([up, gate], dim=1)


def w13_slice(weight_name: str, intermediate_size: int) -> slice:
    if weight_name.endswith(("up_proj.weight", "w3")):
        return slice(0, intermediate_size)
    if weight_name.endswith(("gate_proj.weight", "w1")):
        return slice(intermediate_size, 2 * intermediate_size)
    raise ValueError(f"unknown gate/up weight: {weight_name}")


def normalize_fused_weight(
    loaded: torch.Tensor,
    expected_shape: tuple[int, int, int],
    name: str,
) -> torch.Tensor:
    assert loaded.ndim == 3, f"{name} must be 3D, got {tuple(loaded.shape)}"
    if loaded.shape == expected_shape:
        return loaded
    transposed = (expected_shape[0], expected_shape[2], expected_shape[1])
    if loaded.shape == transposed:
        loaded = loaded.transpose(1, 2).contiguous()
    assert loaded.shape == expected_shape, (
        f"{name} shape {tuple(loaded.shape)} != {expected_shape}"
    )
    return loaded
