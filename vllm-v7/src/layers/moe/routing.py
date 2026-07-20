from __future__ import annotations

import torch
import torch.nn.functional as F


def route_experts(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    norm_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(scores, top_k, dim=-1)
    if norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(router_logits.dtype), topk_ids
