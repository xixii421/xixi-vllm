import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    residual_ptr,
    out_ptr,
    new_residual_ptr,
    weight_ptr,
    x_row_stride,
    residual_row_stride,
    out_row_stride,
    new_residual_row_stride,
    hidden_size,
    eps,
    ADD_RESIDUAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    if ADD_RESIDUAL:
        residual = tl.load(
            residual_ptr + row * residual_row_stride + cols, mask=mask, other=0.0
        ).to(tl.float32)
        x = x + residual
        tl.store(
            new_residual_ptr + row * new_residual_row_stride + cols, x, mask=mask
        )

    rms = tl.rsqrt(tl.sum(x * x, axis=0) / hidden_size + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        out_ptr + row * out_row_stride + cols, x * rms * weight, mask=mask
    )


def _as_norm_rows(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Flatten ``[..., hidden]`` into contiguous rows when needed."""
    hidden = x.shape[-1]
    if x.stride(-1) != 1 or not x.is_contiguous():
        x = x.contiguous()
    rows = x.reshape(-1, hidden)
    return rows, rows.stride(0), rows.shape[0]


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not x.is_cuda:
            return self._forward_native(x, residual)

        shape = x.shape
        hidden_size = shape[-1]
        out = torch.empty(shape, device=x.device, dtype=x.dtype)
        block = triton.next_power_of_2(hidden_size)

        if x.ndim == 3 and x.stride(-1) == 1 and not x.is_contiguous():
            # Strided [T, H, D] from packed QKV: one program per (token, head).
            return self._forward_strided_heads(x, residual, out, block)

        x_2d, x_stride, rows = _as_norm_rows(x)
        out_2d = out.view(rows, hidden_size)

        if residual is None:
            _rms_norm_kernel[(rows,)](
                x_2d,
                None,
                out_2d,
                None,
                self.weight,
                x_stride,
                0,
                out_2d.stride(0),
                0,
                hidden_size,
                self.eps,
                ADD_RESIDUAL=False,
                BLOCK_SIZE=block,
            )
            return out

        if residual.stride(-1) != 1 or (
            residual.ndim == 3 and not residual.is_contiguous()
        ):
            # Residual path is used for hidden states, normally contiguous.
            residual = residual.contiguous()
        res_2d, res_stride, res_rows = _as_norm_rows(residual)
        assert res_rows == rows
        new_residual = torch.empty_like(out)
        new_res_2d = new_residual.view(rows, hidden_size)
        _rms_norm_kernel[(rows,)](
            x_2d,
            res_2d,
            out_2d,
            new_res_2d,
            self.weight,
            x_stride,
            res_stride,
            out_2d.stride(0),
            new_res_2d.stride(0),
            hidden_size,
            self.eps,
            ADD_RESIDUAL=True,
            BLOCK_SIZE=block,
        )
        return out, new_residual

    def _forward_strided_heads(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        out: torch.Tensor,
        block: int,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """RMSNorm for ``[T, H, D]`` with contiguous ``D`` and arbitrary token stride."""
        assert residual is None, "strided Q/K norm does not take residual"
        tokens, num_heads, hidden_size = x.shape
        rows = tokens * num_heads
        # Flattened row pointer: token * stride0 + head * stride1
        # Encode as a single kernel by using a gathered base via arange math.
        # Triton program_id -> (t, h).
        return self._launch_strided_heads(x, out, rows, block)

    def _launch_strided_heads(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
        rows: int,
        block: int,
    ) -> torch.Tensor:
        tokens, num_heads, hidden_size = x.shape
        _rms_norm_strided_heads_kernel[(rows,)](
            x,
            out,
            self.weight,
            x.stride(0),
            x.stride(1),
            out.stride(0),
            out.stride(1),
            num_heads,
            hidden_size,
            self.eps,
            BLOCK_SIZE=block,
        )
        return out

    def _forward_native(
        self, x: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        # Always work on a real fp32 buffer so float32 inputs are not mutated.
        x_float = x.float().clone()
        if residual is not None:
            x_float.add_(residual.float())
            new_residual = x_float.to(dtype)

        x_float.mul_(
            torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        )
        out = x_float.to(dtype).mul_(self.weight)
        return out if residual is None else (out, new_residual)


@triton.jit
def _rms_norm_strided_heads_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    token_stride,
    head_stride,
    out_token_stride,
    out_head_stride,
    num_heads,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // num_heads
    head = row % num_heads
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size
    x_off = token * token_stride + head * head_stride + cols
    x = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
    rms = tl.rsqrt(tl.sum(x * x, axis=0) / hidden_size + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out_off = token * out_token_stride + head * out_head_stride + cols
    tl.store(out_ptr + out_off, x * rms * weight, mask=mask)
