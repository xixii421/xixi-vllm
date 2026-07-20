from __future__ import annotations

from dataclasses import dataclass

import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)

from .attention import AttentionMetadata

_WORKSPACE_BYTES = 256 * 1024 * 1024


@dataclass
class BackendConfig:
    num_heads: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    dtype: torch.dtype
    device: torch.device
    workspace_bytes: int = _WORKSPACE_BYTES


class FlashInferBackend:
    def __init__(self, cfg: BackendConfig) -> None:
        self.cfg = cfg
        self.sm_scale = cfg.head_dim**-0.5

        self.prefill_workspace = torch.empty(
            cfg.workspace_bytes, dtype=torch.uint8, device=cfg.device
        )
        self.decode_workspace = torch.empty(
            cfg.workspace_bytes, dtype=torch.uint8, device=cfg.device
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.prefill_workspace, kv_layout="NHD"
        )
        # Match official vLLM FlashInfer backend: tensor cores are at least as
        # fast as CUDA cores on recent GPUs (esp. GQA).
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.decode_workspace, kv_layout="NHD", use_tensor_cores=True
        )

    def build_metadata(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        slot_mapping: torch.Tensor,
        is_prefill: bool,
        kv_cache: torch.Tensor,
        qo_indptr: torch.Tensor | None = None,
    ) -> AttentionMetadata:
        """Plan the matching FlashInfer wrapper and pack step metadata."""
        c = self.cfg
        if is_prefill:
            assert qo_indptr is not None, "prefill requires qo_indptr"
            self.prefill_wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indices=indices,
                paged_kv_indptr=indptr,
                paged_kv_last_page_len=last_page_len,
                num_qo_heads=c.num_heads,
                num_kv_heads=c.num_kv_heads,
                head_dim_qk=c.head_dim,
                head_dim_vo=c.head_dim,
                page_size=c.block_size,
                causal=True,
                sm_scale=self.sm_scale,
                q_data_type=c.dtype,
                kv_data_type=c.dtype,
            )
            wrapper = self.prefill_wrapper
        else:
            self.decode_wrapper.plan(
                indptr=indptr,
                indices=indices,
                last_page_len=last_page_len,
                num_qo_heads=c.num_heads,
                num_kv_heads=c.num_kv_heads,
                head_dim=c.head_dim,
                page_size=c.block_size,
                sm_scale=self.sm_scale,
                q_data_type=c.dtype,
                kv_data_type=c.dtype,
            )
            wrapper = self.decode_wrapper

        return AttentionMetadata(
            wrapper=wrapper,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
        )


def compute_paged_kv_metadata(
    seq_lens: list[int],
    block_tables: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    n_pages = [(s + block_size - 1) // block_size for s in seq_lens]
    indptr = [0]
    for n in n_pages:
        indptr.append(indptr[-1] + n)

    indptr_cpu = torch.tensor(indptr, dtype=torch.int32, device="cpu")
    last_page_len = torch.tensor(
        [(s - 1) % block_size + 1 for s in seq_lens],
        dtype=torch.int32,
        device="cpu",
    )

    total = indptr[-1]
    if total == 0:
        indices = torch.empty(0, dtype=torch.int32, device=device)
    else:
        # Keep pages ``block_tables[i, :n_pages[i]]``; drop unused padding.
        max_pages = block_tables.size(1)
        page_ids = torch.arange(max_pages, device=device)
        n_pages_t = torch.tensor(n_pages, dtype=torch.int32, device=device)
        mask = page_ids.unsqueeze(0) < n_pages_t.unsqueeze(1)
        indices = block_tables[mask].to(dtype=torch.int32)

    return indptr_cpu, indices, last_page_len
