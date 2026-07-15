from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import nn


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


@triton.jit
def _triton_rope_forward(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    num_tokens,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
):
    pid = tl.program_id(0)
    half_rd = rd // 2

    off_cos = pid * half_rd + tl.arange(0, pad_hd // 2)
    mask_cos = tl.arange(0, pad_hd // 2) < half_rd
    cos_row = tl.load(cos_ptr + off_cos, mask=mask_cos, other=0).to(tl.float32)
    sin_row = tl.load(sin_ptr + off_cos, mask=mask_cos, other=0).to(tl.float32)

    q_base = q_ptr + pid * (n_qh * hd)
    off_q1 = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    off_q2 = off_q1 + half_rd
    mask_q = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
        tl.arange(0, pad_hd // 2)[None, :] < half_rd
    )
    q1 = tl.load(q_base + off_q1, mask=mask_q, other=0).to(tl.float32)
    q2 = tl.load(q_base + off_q2, mask=mask_q, other=0).to(tl.float32)
    tl.store(q_base + off_q1, (q1 * cos_row - q2 * sin_row).to(q1.dtype), mask=mask_q)
    tl.store(q_base + off_q2, (q2 * cos_row + q1 * sin_row).to(q2.dtype), mask=mask_q)

    k_base = k_ptr + pid * (n_kh * hd)
    off_k1 = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    off_k2 = off_k1 + half_rd
    mask_k = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
        tl.arange(0, pad_hd // 2)[None, :] < half_rd
    )
    k1 = tl.load(k_base + off_k1, mask=mask_k, other=0).to(tl.float32)
    k2 = tl.load(k_base + off_k2, mask=mask_k, other=0).to(tl.float32)
    tl.store(k_base + off_k1, (k1 * cos_row - k2 * sin_row).to(k1.dtype), mask=mask_k)
    tl.store(k_base + off_k2, (k2 * cos_row + k1 * sin_row).to(k2.dtype), mask=mask_k)


def _rope_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_size: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, n_qh, hd = q.shape
    n_kh = k.shape[1]
    assert hd == head_size and rotary_dim == head_size

    q_flat = q.reshape(n, n_qh * hd).contiguous()
    k_flat = k.reshape(n, n_kh * hd).contiguous()

    _triton_rope_forward[(n,)](
        q_flat,
        k_flat,
        cos.contiguous(),
        sin.contiguous(),
        n,
        n_qh,
        n_kh,
        hd,
        rotary_dim,
        triton.next_power_of_2(n_qh),
        triton.next_power_of_2(n_kh),
        triton.next_power_of_2(hd),
    )
    return q_flat.view_as(q), k_flat.view_as(k)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        assert rotary_dim == head_size, "v4 RoPE requires rotary_dim == head_size"
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _get_cos_sin(self, positions: torch.Tensor):
        cs = self.cos_sin_cache[positions]
        return cs.chunk(2, dim=-1)

    def _forward_native(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._get_cos_sin(positions)
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)

    def _forward_cuda(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._get_cos_sin(positions)
        return _rope_cuda(
            query,
            key,
            cos.squeeze(1),
            sin.squeeze(1),
            self.head_size,
            self.rotary_dim,
        )

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.is_cuda:
            return self._forward_cuda(positions, query, key)
        return self._forward_native(positions, query, key)
