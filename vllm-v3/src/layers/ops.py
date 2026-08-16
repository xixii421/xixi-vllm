"""torch.compile 自定义算子注册。

该模块将 flashinfer 的底层 kernel 包装为 torch.library 自定义算子，
避免编译器尝试 trace 进这些 kernel 内部。

设计要点：
    - 每个算子通过 torch.library.custom_op 注册，编译时作为黑盒处理。
    - 每个算子注册 fake 实现，供 torch.compile 进行形状和 dtype 推导。
    - 底层 kernel 由 flashinfer 提供，本模块仅做形状适配与接口封装。
"""

from __future__ import annotations
import torch
from flashinfer import apply_rope_with_cos_sin_cache as _flashinfer_rope
from flashinfer.activation import silu_and_mul as _flashinfer_silu_and_mul


@torch.library.custom_op("vllm::silu_and_mul", mutates_args=())
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:

    return _flashinfer_silu_and_mul(x)


@silu_and_mul.register_fake
def _silu_and_mul_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty((x.shape[0], x.shape[1] // 2), dtype=x.dtype, device=x.device)


@torch.library.custom_op("vllm::rotary_embedding", mutates_args=())
def rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """执行 RoPE 旋转。"""
    n, n_qh = q.shape[:2]
    n_kh = k.shape[1]

    # flashinfer 要求输入为 [tokens, num_heads * head_size] 的连续张量。
    qf = q.reshape(n, n_qh * head_size).contiguous()
    kf = k.reshape(n, n_kh * head_size).contiguous()

    # 使用 neox 风格的 RoPE 实现。
    qo, ko = _flashinfer_rope(positions, qf, kf, head_size, cos_sin_cache, is_neox=True)

    # 恢复原始形状后返回。
    return qo.view_as(q), ko.view_as(k)


@rotary_embedding.register_fake
def _rotary_embedding_fake(q, k, positions, cos_sin_cache, head_size):
    return q, k