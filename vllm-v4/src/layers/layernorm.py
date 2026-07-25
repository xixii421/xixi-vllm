import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _rms_norm_fused_kernel(
    X_ptr,
    Residual_ptr,
    Out_ptr,
    New_Res_ptr,
    Weight_ptr,
    row_stride,
    N,
    eps,
    ADD_RESIDUAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row_ptr = X_ptr + row_idx * row_stride
    out_row_ptr = Out_ptr + row_idx * row_stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    if ADD_RESIDUAL:
        res_row_ptr = Residual_ptr + row_idx * row_stride
        new_res_row_ptr = New_Res_ptr + row_idx * row_stride
        residual = tl.load(res_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = x + residual
        tl.store(new_res_row_ptr + cols, x, mask=mask)

    variance = tl.sum(x * x, axis=0) / N
    rsqrt_var = tl.rsqrt(variance + eps)
    weight = tl.load(Weight_ptr + cols, mask=mask).to(tl.float32)
    out = x * rsqrt_var * weight
    tl.store(out_row_ptr + cols, out, mask=mask)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if not x.is_cuda:
            return self._forward_cpu(x, residual)

        orig_shape = x.shape
        hidden_size = orig_shape[-1]
        if not x.is_contiguous():
            x = x.contiguous()
        x_2d = x.view(-1, hidden_size)
        out_2d = torch.empty_like(x_2d)

        grid = (x_2d.shape[0],)
        block = triton.next_power_of_2(hidden_size)

        if residual is None:
            _rms_norm_fused_kernel[grid](
                x_2d,
                x_2d,
                out_2d,
                out_2d,
                self.weight,
                x_2d.stride(0),
                hidden_size,
                self.eps,
                ADD_RESIDUAL=False,
                BLOCK_SIZE=block,
            )
            return out_2d.view(orig_shape)

        if not residual.is_contiguous():
            residual = residual.contiguous()
        residual_2d = residual.view(-1, hidden_size)
        new_residual_2d = torch.empty_like(residual_2d)
        _rms_norm_fused_kernel[grid](
            x_2d,
            residual_2d,
            out_2d,
            new_residual_2d,
            self.weight,
            x_2d.stride(0),
            hidden_size,
            self.eps,
            ADD_RESIDUAL=True,
            BLOCK_SIZE=block,
        )
        return out_2d.view(orig_shape), new_residual_2d.view(orig_shape)

    def _forward_cpu(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ):
        orig_dtype = x.dtype
        if residual is None:
            y = x.float()
            y.mul_(torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps))
            return y.to(orig_dtype).mul_(self.weight)
        y = x.float().add_(residual.float())
        new_res = y.to(orig_dtype)
        y.mul_(torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps))
        return y.to(orig_dtype).mul_(self.weight), new_res
