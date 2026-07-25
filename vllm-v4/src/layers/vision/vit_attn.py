import torch
from flash_attn import flash_attn_varlen_func
from typing import List, Optional

class ViTAttentionBackend:
    """ 基于 FlashAttention 的 ViT 变长序列注意力后端。"""
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
        self.scale = head_dim ** -0.5
        
        # 状态缓存
        self._cu_seqlens: Optional[torch.Tensor] = None
        self._max_seqlen: int = 0

    def plan(self, cu_seqlens_cpu: torch.Tensor, seqlens: Optional[List[int]] = None) -> None:
        """
        准备元数据。
        契约: cu_seqlens_cpu 必须为 CPU Tensor (推荐 pin_memory=True 以实现真异步拷贝)。
        """
        # 异步传输至 GPU 并强制转为 int32
        self._cu_seqlens = cu_seqlens_cpu.to(
            dtype=torch.int32, device=self.device, non_blocking=True
        )
        
        if seqlens:
            self._max_seqlen = max(seqlens)
        else:
            self._max_seqlen = int((cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item())

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        执行变长注意力。
        契约: q, k, v 形状必须为 (total_tokens, num_heads, head_dim)。
        """
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=self._cu_seqlens,
            cu_seqlens_k=self._cu_seqlens,
            max_seqlen_q=self._max_seqlen,
            max_seqlen_k=self._max_seqlen,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=False,
        )