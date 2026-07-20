import torch
import torch.nn.functional as F
from torch import nn

from flashinfer import silu_and_mul as _flashinfer_silu_and_mul


@torch.compile
def _silu_and_mul_native(x: torch.Tensor) -> torch.Tensor:
    gate, value = x.chunk(2, dim=-1)
    return F.silu(gate) * value


def _silu_and_mul_up_gate_native(x: torch.Tensor) -> torch.Tensor:
    up, gate = x.chunk(2, dim=-1)
    return up * F.silu(gate)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Fused SiLU: ``silu(x[..., :I]) * x[..., I:]`` — layout ``[gate | up]``."""
    if x.device.type == "cuda":
        try:
            from .moe.cuda_ops import moe_cuda_available, silu_and_mul_cuda

            if moe_cuda_available():
                out = x.new_empty(*x.shape[:-1], x.shape[-1] // 2)
                silu_and_mul_cuda(out, x.contiguous())
                return out
        except Exception:
            pass
        return _flashinfer_silu_and_mul(x)
    return _silu_and_mul_native(x)


def silu_and_mul_up_gate(x: torch.Tensor) -> torch.Tensor:
    """Cutlass / Triton MoE GEMM1 layout ``[up | gate]`` → ``up * silu(gate)``."""
    if x.device.type == "cuda":
        try:
            from .moe.cuda_ops import moe_cuda_available, mul_and_silu_cuda

            if moe_cuda_available():
                out = x.new_empty(*x.shape[:-1], x.shape[-1] // 2)
                mul_and_silu_cuda(out, x.contiguous())
                return out
        except Exception:
            pass
    return _silu_and_mul_up_gate_native(x)


class SiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul(x)
