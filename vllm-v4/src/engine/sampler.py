from __future__ import annotations

import torch
from torch import nn


def _apply_top_k(logits: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
    vocab = logits.size(-1)
    k = top_k.to(dtype=torch.long, device=logits.device).clamp(max=vocab)
    if bool((k <= 0).all()):
        return logits

    k_eff = torch.where(k <= 0, torch.full_like(k, vocab), k)
    max_k = int(k_eff.max().item())
    topk_vals, _ = torch.topk(logits, max_k, dim=-1)
    # Threshold is the k-th value per row (index k_eff-1).
    gather_idx = (k_eff - 1).clamp(min=0).unsqueeze(-1)
    thresholds = topk_vals.gather(1, gather_idx)
    mask = logits < thresholds
    # Disabled rows: never mask.
    disabled = (k <= 0).unsqueeze(-1)
    logits = logits.masked_fill(mask & ~disabled, float("-inf"))
    return logits


def _apply_top_p(logits: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    p = top_p.to(dtype=torch.float32, device=logits.device)
    if bool((p >= 1.0 - 1e-12).all()):
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = probs.cumsum(dim=-1)
    sorted_mask = cum - probs > p.unsqueeze(-1)
    sorted_mask[..., 0] = False
    disabled = (p >= 1.0 - 1e-12).unsqueeze(-1)
    sorted_mask = sorted_mask & ~disabled
    mask = torch.zeros_like(sorted_mask).scatter(1, sorted_idx, sorted_mask)
    return logits.masked_fill(mask, float("-inf"))


def _apply_min_p(logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
    mp = min_p.to(dtype=torch.float32, device=logits.device)
    if bool((mp <= 0).all()):
        return logits
    probs = torch.softmax(logits.float(), dim=-1)
    max_p = probs.max(dim=-1, keepdim=True).values
    thresh = max_p * mp.unsqueeze(-1)
    mask = probs < thresh
    disabled = (mp <= 0).unsqueeze(-1)
    return logits.masked_fill(mask & ~disabled, float("-inf"))


class Sampler(nn.Module):
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor | None = None,
        top_ks: torch.Tensor | None = None,
        min_ps: torch.Tensor | None = None,
    ) -> torch.Tensor:

        temperatures = temperatures.to(dtype=torch.float32, device=logits.device)
        greedy_mask = temperatures <= 1e-5
        bs = logits.size(0)

        if top_ps is None:
            top_ps = torch.ones(bs, device=logits.device, dtype=torch.float32)
        else:
            top_ps = top_ps.to(dtype=torch.float32, device=logits.device)
            top_ps = torch.where(top_ps < 0, torch.ones_like(top_ps), top_ps)

        if top_ks is None:
            top_ks = torch.zeros(bs, device=logits.device, dtype=torch.long)
        else:
            top_ks = top_ks.to(dtype=torch.long, device=logits.device)

        if min_ps is None:
            min_ps = torch.zeros(bs, device=logits.device, dtype=torch.float32)
        else:
            min_ps = min_ps.to(dtype=torch.float32, device=logits.device)

        if bool(greedy_mask.all()):
            return logits.argmax(dim=-1)

        safe_temp = temperatures.clamp(min=1e-5)
        scaled = logits.float() / safe_temp.unsqueeze(dim=1)
        scaled = _apply_top_k(scaled, top_ks)
        scaled = _apply_top_p(scaled, top_ps)
        scaled = _apply_min_p(scaled, min_ps)

        probs = torch.softmax(scaled, dim=-1)

        probs = torch.nan_to_num(probs, nan=0.0)
        sample = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)

        if bool(greedy_mask.any()):
            greedy = logits.argmax(dim=-1)
            sample = torch.where(greedy_mask, greedy, sample)
        return sample
