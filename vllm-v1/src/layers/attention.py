"""GQA 注意力 + 分页 KV Cache（slot 写入 → 页表读回 → SDPA）。

输入输出布局均为 [B, S, H, D]。forward 只依赖 ForwardContext 暴露的
store / slot_mapping / block_table，与 KVManager 生命周期解耦。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..kv.context import get_forward_context


class Attention(nn.Module):
    """多头注意力"""

    def __init__(
        self,
        num_heads: int,     # Q 头数，如 Qwen3-0.6B 为 16
        head_dim: int,      # 每头向量维度，如 128
        num_kv_heads: int,  # K/V 头数（GQA），如 8；须整除 num_heads
        layer_id: int = 0,  # 本层号，KV Cache 写/读定位物理页用
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.layer_id = layer_id
        # 缩放因子：QK^T / sqrt(head_dim)，构造器预计算避免 forward 重复算
        self.scaling = head_dim**-0.5

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # enable_gqa：K/V 头少于 Q 头时在融合内核内广播，免手写 repeat
        kwargs: dict = {"scale": self.scaling}
        if attn_mask is not None:
            kwargs["attn_mask"] = attn_mask
        else:
            kwargs["is_causal"] = is_causal
        if k.shape[1] != q.shape[1]:
            kwargs["enable_gqa"] = True
        return F.scaled_dot_product_attention(q, k, v, **kwargs)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        ctx = get_forward_context()
        # [B, S, H, D] → [B, H, S, D]（SDPA 要求 head 维在前）
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 无 ForwardContext（如单独跑模型）：走无缓存 SDPA
        if ctx is None:
            out = self._sdpa(q, k, v, is_causal=True)
            return out.transpose(1, 2).contiguous()

        start = ctx.cache_len   # 本步开始前已有 KV 长度
        n = q.shape[2]          # 本步新增 token 数
        total = ctx.seq_len     # 本步写完后的总 KV 长度
        assert total == start + n

        # 1) write：本步 Post-RoPE 的 K/V → 按 slot_mapping 写入页池（仅新增位置）
        ctx.store.write(self.layer_id, ctx.slot_mapping, k.transpose(1, 2), v.transpose(1, 2))

        # 2) gather：按 block_table 读回逻辑长度 total 的全部历史 K/V
        k_c, v_c = ctx.store.gather(self.layer_id, ctx.block_table, total)
        k_c = k_c.unsqueeze(0).transpose(1, 2)  # [1, Hkv, T, D]
        v_c = v_c.unsqueeze(0).transpose(1, 2)

        # 3) attend：本步 Q 与历史做注意力，掩码随 Prefill / Decode / 部分 Prefill 变化
        if start == 0 and n == total:
            out = self._sdpa(q, k_c, v_c, is_causal=True)      # 完整 prefill：下三角
        elif n == 1:
            out = self._sdpa(q, k_c, v_c, is_causal=False)     # decode：只看历史，无掩码
        else:
            # 前缀命中后的部分 Prefill：Q 只覆盖新增区间，需手工下三角掩码
            device, dtype = q.device, q.dtype
            q_pos = torch.arange(start, start + n, device=device)[:, None]
            kv_pos = torch.arange(total, device=device)[None, :]
            attn_mask = torch.zeros(n, total, device=device, dtype=dtype)
            attn_mask.masked_fill_(kv_pos > q_pos, torch.finfo(dtype).min)
            out = self._sdpa(q, k_c, v_c, is_causal=False, attn_mask=attn_mask)

        # 换回 [B, S, H, D]；transpose 产生非连续 view，contiguous() 重排供后续 GEMM
        return out.transpose(1, 2).contiguous()
