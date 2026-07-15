"""Layer unit tests: RoPE, RMSNorm, SiluAndMul."""

from __future__ import annotations

import pytest
import torch

from src.layers.activation import SiluAndMul
from src.layers.layernorm import RMSNorm
from src.layers.rotary import RotaryEmbedding, apply_rotary_emb


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA preferred")


class TestLayers:
    def test_silu_and_mul(self):
        m = SiluAndMul().cuda()
        x = torch.randn(4, 16, device="cuda", dtype=torch.float16)
        # SiluAndMul expects concatenated [gate, up]
        y = m(x)
        assert y.shape == (4, 8)

    def test_rmsnorm(self):
        m = RMSNorm(32, eps=1e-6).cuda().half()
        x = torch.randn(8, 32, device="cuda", dtype=torch.float16)
        y = m(x)
        assert y.shape == x.shape
        # residual form
        y2, r = m(x, x)
        assert y2.shape == x.shape and r.shape == x.shape

    def test_rope_native_vs_cuda(self):
        hd = 64
        rope = RotaryEmbedding(hd, hd, 128, 10000.0).cuda()
        pos = torch.arange(8, device="cuda")
        q = torch.randn(8, 4, hd, device="cuda", dtype=torch.float16)
        k = torch.randn(8, 2, hd, device="cuda", dtype=torch.float16)
        q1, k1 = rope._forward_native(pos, q.clone(), k.clone())
        q2, k2 = rope._forward_cuda(pos, q.clone(), k.clone())
        assert torch.allclose(q1, q2, atol=2e-2, rtol=2e-2)
        assert torch.allclose(k1, k2, atol=2e-2, rtol=2e-2)

    def test_apply_rotary_emb_unit(self):
        x = torch.randn(3, 8)
        cos = torch.ones(3, 4)
        sin = torch.zeros(3, 4)
        y = apply_rotary_emb(x, cos, sin)
        # sin=0 → y1=x1*cos, y2=x2*cos
        x1, x2 = x.float().chunk(2, dim=-1)
        assert torch.allclose(y.float(), torch.cat([x1, x2], dim=-1), atol=1e-5)
