"""GQA attention with PagedAttention storage (slot write → gather → SDPA)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..kv.context import get_forward_context


def _sdpa_supports_gqa() -> bool:
    try:
        q = torch.zeros(1, 2, 1, 4)
        k = torch.zeros(1, 1, 1, 4)
        F.scaled_dot_product_attention(q, k, k, is_causal=True, enable_gqa=True)
        return True
    except TypeError:
        return False


_SDPA_HAS_GQA = _sdpa_supports_gqa()


class Attention(nn.Module):
    """Input/output layout ``[B, S, H, D]``; SDPA uses ``[B, H, S, D]``.

    Couples only to ``ForwardContext``'s PagedAttention fields
    (``store`` / ``slot_mapping`` / ``block_table``), not to ``KVManager``.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int = 0,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.layer_id = layer_id
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = scaling

    def _expand_gqa(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_kv_groups <= 1:
            return k, v
        b, hkv, s, d = k.shape
        k = k.unsqueeze(2).expand(b, hkv, self.num_kv_groups, s, d).reshape(
            b, self.num_heads, s, d
        )
        v = v.unsqueeze(2).expand(b, hkv, self.num_kv_groups, s, d).reshape(
            b, self.num_heads, s, d
        )
        return k, v

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs: dict = {"scale": self.scaling}
        if attn_mask is not None:
            kwargs["attn_mask"] = attn_mask
        else:
            kwargs["is_causal"] = is_causal
        if _SDPA_HAS_GQA and k.shape[1] != q.shape[1]:
            kwargs["enable_gqa"] = True
            return F.scaled_dot_product_attention(q, k, v, **kwargs)
        if k.shape[1] != q.shape[1]:
            k, v = self._expand_gqa(k, v)
        return F.scaled_dot_product_attention(q, k, v, **kwargs)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        ctx = get_forward_context()
        q = q.transpose(1, 2)  # [B,H,S,D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if ctx is None:
            out = self._sdpa(q, k, v, is_causal=True)
            return out.transpose(1, 2).contiguous()

        start = ctx.cache_len
        n = q.shape[2]
        # Post-RoPE K/V → physical pages via slot_mapping.
        ctx.store.write(
            self.layer_id,
            ctx.slot_mapping,
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        total = ctx.seq_len
        assert total == start + n
        k_c, v_c = ctx.store.gather(self.layer_id, ctx.block_table, total)
        k_c = k_c.unsqueeze(0).transpose(1, 2)  # [1,Hkv,T,D]
        v_c = v_c.unsqueeze(0).transpose(1, 2)

        if start == 0 and n == total:
            out = self._sdpa(q, k_c, v_c, is_causal=True)
        elif n == 1:
            out = self._sdpa(q, k_c, v_c, is_causal=False)
        else:
            # Partial prefill after a prefix hit: Q is only the new span.
            device, dtype = q.device, q.dtype
            q_pos = torch.arange(start, start + n, device=device)[:, None]
            kv_pos = torch.arange(total, device=device)[None, :]
            keep = kv_pos <= q_pos
            neg = torch.finfo(dtype).min
            attn_mask = torch.zeros(n, total, device=device, dtype=dtype)
            attn_mask = attn_mask.masked_fill(~keep, neg)
            out = self._sdpa(q, k_c, v_c, is_causal=False, attn_mask=attn_mask)

        return out.transpose(1, 2).contiguous()
