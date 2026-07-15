import torch
from torch import nn


class Sampler(nn.Module):
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor | None = None,
        min_ps: torch.Tensor | None = None,
        top_ps: torch.Tensor | None = None,
        top_ks: torch.Tensor | None = None,
    ):
        is_all_greedy = (
            temperatures is None and min_ps is None and top_ps is None and top_ks is None
        )
        if is_all_greedy:
            return torch.argmax(logits, dim=-1)

        logits = logits.to(torch.float)
        if temperatures is not None:
            temperatures = torch.where(temperatures <= 0, 1.0, temperatures)
            logits.div_(temperatures.unsqueeze(dim=1))
        logits = _apply_top_k_top_p(logits, top_ps, top_ks)
        logits = _apply_min_p(logits, min_ps)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        return torch.multinomial(probs, num_samples=1).squeeze(dim=-1)


def _apply_top_k_top_p(logits, p=None, k=None):
    if p is None and k is None:
        return logits
    vocab_size = logits.size(-1)
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        k = torch.where(k <= 0, vocab_size, k)
        top_k_mask = logits_sort.size(1) - k.to(torch.long)
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        logits_sort.masked_fill_(logits_sort < top_k_mask, -float("inf"))

    if p is not None:
        p = torch.where(p <= 0, 1.0, p)
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = probs_sort.cumsum(dim=-1)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    return torch.empty_like(logits_sort).scatter_(dim=-1, index=logits_idx, src=logits_sort)


def _apply_min_p(logits, min_p=None):
    if min_p is None:
        return logits
    probs = torch.softmax(logits, dim=-1)
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    return logits.masked_fill_(probs < min_p.unsqueeze_(dim=1) * top_probs, -float("inf"))
