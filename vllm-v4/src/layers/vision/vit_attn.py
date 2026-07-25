"""ViT attention via flash-attn varlen (supports Qwen3-VL head_dim=72).

LM paged decode/prefill stays on FlashInfer (block tables + Decode CUDA Graph);
flash-attn does not replace that path without a full KV rewrite.
"""

from __future__ import annotations

import torch
from flash_attn import flash_attn_varlen_func


class ViTAttentionBackend:
    """Packed ragged self-attention: q/k/v ``[N, H, D]`` + ``cu_seqlens``."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        **_kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.scale = head_dim**-0.5
        self._cu: torch.Tensor | None = None
        self._max_seqlen = 0

    def plan(self, cu_seqlens_cpu: torch.Tensor, seqlens: list[int] | None = None) -> None:
        # flash-attn wants int32 cu_seqlens on the same device as q.
        self._cu = cu_seqlens_cpu.to(dtype=torch.int32, device=self.device, non_blocking=True)
        if seqlens:
            self._max_seqlen = max(seqlens) if seqlens else 0
        else:
            diffs = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
            self._max_seqlen = int(diffs.max().item()) if diffs.numel() else 0

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        assert self._cu is not None, "plan() before run()"
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=self._cu,
            cu_seqlens_k=self._cu,
            max_seqlen_q=self._max_seqlen,
            max_seqlen_k=self._max_seqlen,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=False,
        )
