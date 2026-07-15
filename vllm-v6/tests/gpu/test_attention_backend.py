"""FlashInfer paged metadata + attention backend smoke."""

from __future__ import annotations

import pytest
import torch

from src.layers.attention_backend import (
    BackendConfig,
    FlashInferBackend,
    compute_paged_kv_metadata,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class TestAttentionBackend:
    def test_compute_paged_kv_metadata(self):
        device = torch.device("cuda")
        block_tables = torch.tensor([[0, 1, -1], [2, -1, -1]], dtype=torch.int32, device=device)
        indptr, indices, last = compute_paged_kv_metadata(
            [20, 5], block_tables, block_size=16, device=device
        )
        assert indptr.tolist() == [0, 2, 3]
        assert indices.tolist() == [0, 1, 2]
        # (20-1)%16+1=4 ; (5-1)%16+1=5
        assert last.tolist() == [4, 5]

    def test_backend_plan_prefill_decode(self):
        flashinfer = pytest.importorskip("flashinfer")
        device = torch.device("cuda")
        cfg = BackendConfig(
            num_qo_heads=4,
            num_kv_heads=2,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
            device=device,
            workspace_bytes=32 * 1024 * 1024,
        )
        backend = FlashInferBackend(cfg)
        assert backend.prefill_workspace.data_ptr() != backend.decode_workspace.data_ptr()

        pool = torch.zeros(2, 1, 4, 16, 2, 64, dtype=torch.float16, device=device)
        bt = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
        indptr, indices, last = compute_paged_kv_metadata([20], bt, 16, device)
        sm = torch.arange(20, dtype=torch.int32, device=device)
        cu = torch.tensor([0, 20], dtype=torch.int32)
        meta = backend.build_metadata(1, indptr, indices, last, sm, True, pool, cu)
        assert meta.wrapper is backend.prefill_wrapper

        indptr2, indices2, last2 = compute_paged_kv_metadata([20], bt, 16, device)
        sm2 = torch.tensor([19], dtype=torch.int32, device=device)
        meta2 = backend.build_metadata(1, indptr2, indices2, last2, sm2, False, pool, None)
        assert meta2.wrapper is backend.decode_wrapper
