from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)

if TYPE_CHECKING:
    from .attention import AttentionMetadata


@dataclass
class BackendConfig:
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
    def __init__(self, cfg: BackendConfig) -> None:
        self.cfg = cfg
        self.prefill_workspace = torch.empty(
            cfg.workspace_bytes, dtype=torch.uint8, device=cfg.device
        )
        self.decode_workspace = torch.empty(
            cfg.workspace_bytes, dtype=torch.uint8, device=cfg.device
        )

        self.workspace_buffer = self.decode_workspace
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
        slot_mapping: torch.Tensor,
        is_prefill: bool,
        kv_cache_pool: torch.Tensor,
        cu_seqlen_q_cpu: Optional[torch.Tensor] = None,
    ) -> "AttentionMetadata":
        from .attention import AttentionMetadata

        c = self.cfg
        sm = c.head_dim**-0.5
        if is_prefill:
            assert cu_seqlen_q_cpu is not None
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
            slot_mapping=slot_mapping,
        )


def compute_paged_kv_metadata(
    seq_lens: list[int],
    block_tables: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_blocks = [(s + block_size - 1) // block_size for s in seq_lens]
    indptr = [0]
    for nb in n_blocks:
        indptr.append(indptr[-1] + nb)

    indptr_cpu = torch.tensor(indptr, dtype=torch.int32, device="cpu")
    last_page_len_cpu = torch.tensor(
        [(s - 1) % block_size + 1 for s in seq_lens],
        dtype=torch.int32,
        device="cpu",
    )

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
