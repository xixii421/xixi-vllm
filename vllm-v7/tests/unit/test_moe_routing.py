"""MoE router top-k (CPU)."""

from __future__ import annotations

import torch

from src.layers.moe.routing import route_experts


def test_route_experts_renorm():
    logits = torch.tensor([[1.0, 2.0, 0.5, -1.0]])
    w, ids = route_experts(logits, top_k=2, norm_topk_prob=True)
    assert ids.shape == (1, 2)
    assert torch.allclose(w.sum(-1), torch.ones(1), atol=1e-5)


def test_route_experts_topk_ids():
    logits = torch.tensor([[0.1, 0.2, 0.9, 0.3]])
    _, ids = route_experts(logits, top_k=2, norm_topk_prob=False)
    assert ids[0, 0].item() == 2
