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
        # write expects [B, S, H, D]
        cache.write(
            layer_id,
            ctx.slot_mapping,
            k.transpose(1, 2),
            v.transpose(1, 2),
        )

        if ctx.is_prefill:
            # Prefill: causal attn over this step's tokens.
            k, v = self._expand_gqa(k, v)
            attn_output = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.scaling
            )
        else:
            # Decode: gather full logical prefix from paged blocks.
            k_cached, v_cached = cache.gather(layer_id, ctx.block_table, ctx.seq_len)
            k_cached = k_cached.unsqueeze(0).transpose(1, 2)
            v_cached = v_cached.unsqueeze(0).transpose(1, 2)
            k_cached, v_cached = self._expand_gqa(k_cached, v_cached)
            attn_output = F.scaled_dot_product_attention(
                q, k_cached, v_cached, is_causal=False, scale=self.scaling
            )

        return attn_output.transpose(1, 2).contiguous()
