"""FlashInfer attention 后端。

该模块负责管理 FlashInfer 分页 KV wrapper 的生命周期，并将批次元数据
规划为 attention 层可直接使用的 AttentionMetadata 执行上下文。

职责边界：
    - 本层仅负责 FlashInfer 相关逻辑，包括 workspace 分配、wrapper 创建、
      plan 调用以及分页索引构建。
    - attention 层不依赖本层内部状态，只消费 AttentionMetadata。
    - compute_paged_kv_metadata 是独立于 backend 实例状态的纯数据转换函数，
      供 model_runner 单独调用。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from .attention import AttentionMetadata
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)


@dataclass
class BackendConfig:
    """FlashInfer 后端配置。

    Attributes:
        num_qo_heads: query / output head 数量。
        num_kv_heads: key / value head 数量。
        head_dim: 每个 head 的维度。
        block_size: KV cache 分页块大小。
        dtype: 计算精度。
        device: 目标 GPU 设备。
        workspace_bytes: FlashInfer workspace 字节数。
        use_tensor_cores: decode 是否启用 tensor cores。
        non_blocking: plan 与拷贝是否使用非阻塞模式。
    """

    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    dtype: torch.dtype
    device: torch.device
    workspace_bytes: int = 256 * 1024 * 1024
    use_tensor_cores: bool = True
    non_blocking: bool = True


class FlashInferBackend:
    """FlashInfer attention 后端。

    维护 prefill 与 decode 两套独立的 workspace 和 wrapper，
    并根据批次元数据规划 wrapper，最终构造 AttentionMetadata。
    """

    def __init__(self, cfg: BackendConfig) -> None:
        self.cfg = cfg

        # prefill 与 decode 各自持有独立 workspace，避免相互覆盖。
        self.prefill_workspace = torch.empty(
            cfg.workspace_bytes, dtype=torch.uint8, device=cfg.device
        )
        self.decode_workspace = torch.empty(
            cfg.workspace_bytes, dtype=torch.uint8, device=cfg.device
        )

        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.prefill_workspace, kv_layout="NHD"
        )
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.decode_workspace,
            kv_layout="NHD",
            use_tensor_cores=cfg.use_tensor_cores,
        )

    def build_metadata(
        self,
        batch_bs: int,
        paged_kv_indptr_cpu: torch.Tensor,
        paged_kv_indices_gpu: torch.Tensor,
        paged_kv_last_page_len_cpu: torch.Tensor,
        is_prefill: bool,
        kv_cache_pool: torch.Tensor,
        cu_seqlen_q_cpu: Optional[torch.Tensor] = None,
    ) -> AttentionMetadata:
        """规划对应阶段的 wrapper，并构造 AttentionMetadata。

        Args:
            batch_bs: 批次大小。
            paged_kv_indptr_cpu: CPU 端分页 KV 累计索引边界。
            paged_kv_indices_gpu: GPU 端展平后的物理页索引。
            paged_kv_last_page_len_cpu: CPU 端每个序列最后一页的有效长度。
            is_prefill: True 走 prefill wrapper，False 走 decode wrapper。
            kv_cache_pool: KV cache 物理池张量。
            cu_seqlen_q_cpu: prefill 时必填，query token 的累计边界。

        Returns:
            可供 attention 层使用的 AttentionMetadata。
        """
        c = self.cfg

        # softmax 缩放因子。
        sm = c.head_dim**-0.5

        if is_prefill:
            assert cu_seqlen_q_cpu is not None

            # prefill 需要额外提供 query 累计边界，并启用 causal mask。
            self.prefill_wrapper.plan(
                qo_indptr=cu_seqlen_q_cpu,
                paged_kv_indices=paged_kv_indices_gpu,
                paged_kv_indptr=paged_kv_indptr_cpu,
                paged_kv_last_page_len=paged_kv_last_page_len_cpu,
                num_qo_heads=c.num_qo_heads,
                num_kv_heads=c.num_kv_heads,
                head_dim_qk=c.head_dim,
                head_dim_vo=c.head_dim,
                page_size=c.block_size,
                causal=True,
                sm_scale=sm,
                q_data_type=c.dtype,
                kv_data_type=c.dtype,
                non_blocking=c.non_blocking,
            )
            wrapper = self.prefill_wrapper
        else:
            self.decode_wrapper.plan(
                indptr=paged_kv_indptr_cpu,
                indices=paged_kv_indices_gpu,
                last_page_len=paged_kv_last_page_len_cpu,
                num_qo_heads=c.num_qo_heads,
                num_kv_heads=c.num_kv_heads,
                head_dim=c.head_dim,
                page_size=c.block_size,
                sm_scale=sm,
                q_data_type=c.dtype,
                kv_data_type=c.dtype,
                non_blocking=c.non_blocking,
            )
            wrapper = self.decode_wrapper

        return AttentionMetadata(
            wrapper=wrapper,
            kv_cache_pool=kv_cache_pool,
        )


def compute_paged_kv_metadata(
    seq_lens: list[int],
    block_tables: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将序列级表示转换为 FlashInfer 分页 KV 索引。

    返回三个张量，供 FlashInfer wrapper 的 plan 调用使用：
        - indptr: CPU 端累计页边界。
        - indices: GPU 端展平后的物理页索引。
        - last_page_len: CPU 端每个序列最后一页的有效 token 数。

    Args:
        seq_lens: 每个序列的 KV 长度。
        block_tables: 补齐后的物理页表。
        block_size: KV cache 分页块大小。
        device: 目标 GPU 设备。

    Returns:
        tuple:
            - indptr_cpu: CPU 端累计页边界。
            - indices_gpu: GPU 端展平物理页索引。
            - last_page_len_cpu: CPU 端每序列最后一页长度。
    """
    # 每个序列所需的物理页数量。
    n_blocks = [(s + block_size - 1) // block_size for s in seq_lens]

    # 构建 CSR 风格的累计页边界。
    indptr = [0]
    for nb in n_blocks:
        indptr.append(indptr[-1] + nb)

    indptr_cpu = torch.tensor(indptr, dtype=torch.int32, device="cpu")

    # 每个序列最后一页中有效 token 数量，范围为 1 到 block_size。
    last_page_len_cpu = torch.tensor(
        [(s - 1) % block_size + 1 for s in seq_lens],
        dtype=torch.int32,
        device="cpu",
    )

    # 从补齐后的二维 block_tables 中按有效页掩码提取展平索引。
    total = indptr[-1]
    if total > 0 and block_tables is not None:
        max_blocks = block_tables.shape[1]
        flat_mask = torch.arange(max_blocks, device=device)
        n_blocks_gpu = torch.tensor(n_blocks, dtype=torch.int32, device=device)
        mask = flat_mask.unsqueeze(0) < n_blocks_gpu.unsqueeze(1)
        indices_gpu = block_tables[mask]
    else:
        indices_gpu = torch.empty(0, dtype=torch.int32, device=device)

    return indptr_cpu, indices_gpu, last_page_len_cpu