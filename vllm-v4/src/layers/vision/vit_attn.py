from __future__ import annotations

import torch

try:
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

    HAS_VIT_FLASHINFER = True
except Exception:  # pragma: no cover - optional dependency
    HAS_VIT_FLASHINFER = False


class ViTAttentionBackend:
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        workspace_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        if not HAS_VIT_FLASHINFER:
            raise RuntimeError("FlashInfer not available for ViT attention")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.scale = head_dim**-0.5
        self._workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        self.wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self._workspace, kv_layout="NHD"
        )
        self._planned = False

    def plan(self, cu_seqlens_cpu: torch.Tensor) -> None:
        indptr = cu_seqlens_cpu.to(dtype=torch.int32, device="cpu")
        self.wrapper.plan(
            qo_indptr=indptr,
            kv_indptr=indptr,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim_qk=self.head_dim,
            head_dim_vo=self.head_dim,
            causal=False,
            sm_scale=self.scale,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )
        self._planned = True

    def run(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        assert self._planned, "ViTAttentionBackend.run() before plan()"
        return self.wrapper.run(q, k, v)
