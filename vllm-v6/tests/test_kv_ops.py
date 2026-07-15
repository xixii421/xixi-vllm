"""Triton store_kvcache vs PyTorch reference."""

from __future__ import annotations

import pytest
import torch

from src.scheduling.kv_ops import store_kvcache


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _store_ref(key, value, k_cache, v_cache, slot_mapping):
    n, nh, hd = key.shape
    d = nh * hd
    for i in range(n):
        slot = int(slot_mapping[i].item())
        if slot < 0:
            continue
        k_cache.view(-1, d)[slot] = key[i].reshape(d)
        v_cache.view(-1, d)[slot] = value[i].reshape(d)


class TestStoreKV:
    def test_matches_pytorch_with_padding(self):
        device = "cuda"
        dtype = torch.float16
        n, nh, hd = 128, 8, 64
        num_blocks, block_size = 32, 16
        total = num_blocks * block_size

        key = torch.randn(n, nh, hd, dtype=dtype, device=device)
        value = torch.randn(n, nh, hd, dtype=dtype, device=device)
        slots = torch.randperm(total, dtype=torch.int32, device=device)[:n].clone()
        slots[torch.rand(n, device=device) < 0.2] = -1

        k_pt = torch.zeros(total, nh, hd, dtype=dtype, device=device)
        v_pt = torch.zeros_like(k_pt)
        k_tr = torch.zeros_like(k_pt)
        v_tr = torch.zeros_like(k_pt)

        _store_ref(key, value, k_pt, v_pt, slots)
        store_kvcache(key, value, k_tr, v_tr, slots)

        assert torch.allclose(k_pt, k_tr, atol=1e-3, rtol=1e-3)
        assert torch.allclose(v_pt, v_tr, atol=1e-3, rtol=1e-3)

    def test_skip_neg_one(self):
        device = "cuda"
        key = torch.ones(2, 2, 4, device=device, dtype=torch.float16)
        value = torch.ones_like(key) * 2
        k_c = torch.zeros(16, 2, 4, device=device, dtype=torch.float16)
        v_c = torch.zeros_like(k_c)
        slots = torch.tensor([-1, 3], dtype=torch.int32, device=device)
        store_kvcache(key, value, k_c, v_c, slots)
        assert float(k_c.view(-1, 8)[0].sum()) == 0.0
        assert float(k_c.view(-1, 8)[3].sum()) > 0
