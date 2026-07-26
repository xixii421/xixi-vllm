"""ViT attention backend (FlashAttention varlen), CUDA-graph friendly."""

from __future__ import annotations

from typing import List, Optional

import torch
from flash_attn import flash_attn_varlen_func


class ViTAttentionBackend:
    """FlashAttention-varlen backend for Qwen3-VL ViT.

    For CUDA Graph capture/replay, bind a *static* GPU ``cu_seqlens`` buffer via
    :meth:`bind` so the graph does not capture fresh H2D allocations from
    :meth:`plan`.
    """

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
        self._cu_seqlens: Optional[torch.Tensor] = None
        self._max_seqlen: int = 0

    def plan(self, cu_seqlens_cpu: torch.Tensor, seqlens: Optional[List[int]] = None) -> None:
        """Eager path: H2D copy cu_seqlens (may allocate; not CG-safe)."""
        self._cu_seqlens = cu_seqlens_cpu.to(
            dtype=torch.int32, device=self.device, non_blocking=True
        )
        if seqlens:
            self._max_seqlen = max(seqlens) if seqlens else 0
        else:
            self._max_seqlen = int((cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item())

    def bind(self, cu_seqlens_gpu: torch.Tensor, max_seqlen: int) -> None:
        """CG path: use a preallocated GPU buffer (same storage across replays)."""
        assert cu_seqlens_gpu.device.type == "cuda"
        assert cu_seqlens_gpu.dtype == torch.int32
        self._cu_seqlens = cu_seqlens_gpu
        self._max_seqlen = int(max_seqlen)

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        assert self._cu_seqlens is not None
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=self._cu_seqlens,
            cu_seqlens_k=self._cu_seqlens,
            max_seqlen_q=self._max_seqlen,
            max_seqlen_k=self._max_seqlen,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=False,
        )
