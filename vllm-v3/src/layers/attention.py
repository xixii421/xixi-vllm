from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..utils import get_forward_context


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.layer_id = layer_id

        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = scaling

    def _expand_gqa(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_kv_groups <= 1:
            return k, v
        batch, kv_heads, seq, head_dim = k.shape
        k = k.unsqueeze(2).expand(batch, kv_heads, self.num_kv_groups, seq, head_dim)
        k = k.reshape(batch, self.num_heads, seq, head_dim)
        v = v.unsqueeze(2).expand(batch, kv_heads, self.num_kv_groups, seq, head_dim)
        v = v.reshape(batch, self.num_heads, seq, head_dim)
        return k, v

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        ctx = get_forward_context()
        # [B, S, H, D] → [B, H, S, D] for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if ctx is None or ctx.kv_cache is None:
            k, v = self._expand_gqa(k, v)
            attn_output = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.scaling
            )
            return attn_output.transpose(1, 2).contiguous()

        cache = ctx.kv_cache
        layer_id = self.layer_id
        cache.write(layer_id, ctx.slot_mapping, k.transpose(1, 2), v.transpose(1, 2))

        if ctx.is_prefill:
            return self._prefill_attn(q, k, v, ctx, cache, layer_id)
        return self._decode_attn(q, ctx, cache, layer_id)

    def _prefill_attn(self, q, k, v, ctx, cache, layer_id):
        """
        Prefill batch size is always 1 in the teaching runner.

        If prefix cache hit (cached_lens[0] > 0): Q is the suffix;
        gather full context K/V and apply a shifted causal mask.
        """
        cached = ctx.cached_lens[0] if ctx.cached_lens else 0
        ctx_len = ctx.context_lens[0]
        q_len = ctx.query_lens[0]

        if cached == 0:
            k, v = self._expand_gqa(k, v)
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.scaling
            )
            return out.transpose(1, 2).contiguous()

        # Suffix Q attending into [prefix | new].
        k_full, v_full = cache.gather(layer_id, ctx.block_tables[0], ctx_len)
        k_full = k_full.unsqueeze(0).transpose(1, 2)  # [1, Hkv, S, D]
        v_full = v_full.unsqueeze(0).transpose(1, 2)
        k_full, v_full = self._expand_gqa(k_full, v_full)

        # mask[i, j] True = allowed; query i (abs pos = cached+i) sees j <= cached+i
        device = q.device
        qi = torch.arange(q_len, device=device).unsqueeze(1) + cached
        kj = torch.arange(ctx_len, device=device).unsqueeze(0)
        allow = qi >= kj
        # SDPA expects float additive mask: 0 keep, -inf mask
        attn_mask = torch.zeros(q_len, ctx_len, device=device, dtype=q.dtype)
        attn_mask = attn_mask.masked_fill(~allow, torch.finfo(q.dtype).min)
        out = F.scaled_dot_product_attention(
            q, k_full, v_full, attn_mask=attn_mask, scale=self.scaling
        )
        return out.transpose(1, 2).contiguous()

    def _decode_attn(self, q, ctx, cache, layer_id):
        """Decode: B queries of length 1; gather each prefix and SDPA (loop)."""
        batch = q.shape[0]
        outs = []
        for b in range(batch):
            ctx_len = ctx.context_lens[b]
            k_c, v_c = cache.gather(layer_id, ctx.block_tables[b], ctx_len)
            k_c = k_c.unsqueeze(0).transpose(1, 2)
            v_c = v_c.unsqueeze(0).transpose(1, 2)
            k_c, v_c = self._expand_gqa(k_c, v_c)
            outs.append(
                F.scaled_dot_product_attention(
                    q[b : b + 1], k_c, v_c, is_causal=False, scale=self.scaling
                )
            )
        return torch.cat(outs, dim=0).transpose(1, 2).contiguous()
