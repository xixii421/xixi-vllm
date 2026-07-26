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
def _rope_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    PAD_Q_HEADS: tl.constexpr,
    PAD_KV_HEADS: tl.constexpr,
    PAD_HEAD_DIM: tl.constexpr,
):
    token = tl.program_id(0)
    half_dim = head_dim // 2
    dims = tl.arange(0, PAD_HEAD_DIM // 2)
    freq_offsets = token * half_dim + dims
    freq_mask = dims < half_dim
    cos = tl.load(cos_ptr + freq_offsets, mask=freq_mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + freq_offsets, mask=freq_mask, other=0.0).to(tl.float32)

    q_heads = tl.arange(0, PAD_Q_HEADS)[:, None]
    q_offsets = q_heads * head_dim + dims[None, :]
    q_mask = (q_heads < num_q_heads) & (dims[None, :] < half_dim)
    q_base = q_ptr + token * num_q_heads * head_dim
    q1 = tl.load(q_base + q_offsets, mask=q_mask, other=0.0).to(tl.float32)
    q2 = tl.load(q_base + q_offsets + half_dim, mask=q_mask, other=0.0).to(
        tl.float32
    )
    tl.store(q_base + q_offsets, q1 * cos - q2 * sin, mask=q_mask)
    tl.store(q_base + q_offsets + half_dim, q2 * cos + q1 * sin, mask=q_mask)

    kv_heads = tl.arange(0, PAD_KV_HEADS)[:, None]
    k_offsets = kv_heads * head_dim + dims[None, :]
    k_mask = (kv_heads < num_kv_heads) & (dims[None, :] < half_dim)
    k_base = k_ptr + token * num_kv_heads * head_dim
    k1 = tl.load(k_base + k_offsets, mask=k_mask, other=0.0).to(tl.float32)
    k2 = tl.load(k_base + k_offsets + half_dim, mask=k_mask, other=0.0).to(
        tl.float32
    )
    tl.store(k_base + k_offsets, k1 * cos - k2 * sin, mask=k_mask)
    tl.store(k_base + k_offsets + half_dim, k2 * cos + k1 * sin, mask=k_mask)


def _rope_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_size: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    assert head_dim == head_size == rotary_dim

    q_out = q.reshape(num_tokens, -1).contiguous()
    k_out = k.reshape(num_tokens, -1).contiguous()
    _rope_kernel[(num_tokens,)](
        q_out,
        k_out,
        cos.contiguous(),
        sin.contiguous(),
        num_q_heads,
        num_kv_heads,
        head_dim,
        PAD_Q_HEADS=triton.next_power_of_2(num_q_heads),
        PAD_KV_HEADS=triton.next_power_of_2(num_kv_heads),
        PAD_HEAD_DIM=triton.next_power_of_2(head_dim),
    )
    return q_out.view_as(q), k_out.view_as(k)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        *,
        build_cache: bool = True,
    ) -> None:
        super().__init__()
        assert rotary_dim == head_size, "only full-head RoPE is supported"
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        inv_freq = base ** (
            -torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        if build_cache:
            positions = torch.arange(max_position_embeddings, dtype=torch.float32)
            freqs = torch.outer(positions, inv_freq)
            cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze_(1)
        else:
            cache = torch.empty(0, 1, rotary_dim, dtype=torch.float32)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _get_cos_sin(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_sin_cache[positions].chunk(2, dim=-1)

    def prepare(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._get_cos_sin(positions)
        return cos.contiguous(), sin.contiguous()

    def apply(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.is_cuda:
            return _rope_cuda(
                query,
                key,
                cos.squeeze(1),
                sin.squeeze(1),
                self.head_size,
                self.rotary_dim,
            )
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)

    def _forward_native(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.prepare(positions)
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)

    def _forward_cuda(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.prepare(positions)
        return self.apply(query, key, cos, sin)

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.prepare(positions)
        return self.apply(query, key, cos, sin)
