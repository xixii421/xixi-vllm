import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)

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
        self.prefill_workspace = torch.empty(cfg.workspace_bytes, dtype=torch.uint8, device=cfg.device)
        self.decode_workspace = torch.empty(cfg.workspace_bytes, dtype=torch.uint8, device=cfg.device)

        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.prefill_workspace, kv_layout="NHD"
        )
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.decode_workspace,
            kv_layout="NHD",
            use_tensor_cores=cfg.use_tensor_cores,
        )
        self._sm_scale = cfg.head_dim ** -0.5

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
        layer_id: int = 0,
    ) -> "AttentionMetadata":
        c = self.cfg
        
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
                sm_scale=self._sm_scale,
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
                sm_scale=self._sm_scale,
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
    bs = len(seq_lens)
    
    # 1. 计算每个 sequence 的 block 数量 
    n_blocks = np.array([(s + block_size - 1) // block_size for s in seq_lens], dtype=np.int32)
    
    # 2. 计算 indptr
    indptr_np = np.zeros(bs + 1, dtype=np.int32)
    indptr_np[1:] = np.cumsum(n_blocks)
    indptr_cpu = torch.from_numpy(indptr_np) # 零拷贝 (如果上游是 pinned memory) 或极快拷贝
    
    # 3. 计算 last_page_len
    last_page_len_np = ((np.array(seq_lens, dtype=np.int32) - 1) % block_size) + 1
    last_page_len_cpu = torch.from_numpy(last_page_len_np)
    
    # 4. 展平 block_tables
    total_blocks = int(indptr_np[-1])
    if total_blocks > 0:
        # 确保在 CPU 端进行索引操作 
        bt_np = block_tables.cpu().numpy() if block_tables.is_cuda else block_tables.numpy()
        
        # 使用列表推导式 + concatenate，这是 NumPy 中处理不规则行展平最快的方法
        indices_list = [bt_np[i, :nb] for i, nb in enumerate(n_blocks)]
        indices_np = np.concatenate(indices_list) if indices_list else np.array([], dtype=np.int32)
        
        # 异步传输到 GPU
        indices_gpu = torch.from_numpy(indices_np).to(device, non_blocking=True)
    else:
        indices_gpu = torch.empty(0, dtype=torch.int32, device=device)

    return indptr_cpu, indices_gpu, last_page_len_cpu