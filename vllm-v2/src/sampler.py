"""logits → 下一个 token id：贪心，或 temperature + top-k / top-p / min-p。"""

from __future__ import annotations

import torch


class Sampler:
    """无参数的后处理模块。

    全部采样超参为 None 时走 argmax；否则按
    temperature → top-k/top-p → min-p → softmax → multinomial。
    """

    def __call__(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor | None = None,
        min_ps: torch.Tensor | None = None,
        top_ps: torch.Tensor | None = None,
        top_ks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 快路径：不做过滤、不做 softmax，直接取最大分
        if temperatures is None and min_ps is None and top_ps is None and top_ks is None:
            return torch.argmax(logits, dim=-1)

        # 采样阶段统一 fp32，避免大词表 softmax 在半精度下不稳定
        logits = logits.float()
        if temperatures is not None:
            temperatures = torch.where(temperatures <= 0, 1.0, temperatures)
            logits = logits / temperatures.unsqueeze(1)
        logits = _top_k_top_p(logits, top_ps, top_ks)
        logits = _min_p(logits, min_ps)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _top_k_top_p(
    logits: torch.Tensor,
    p: torch.Tensor | None,
    k: torch.Tensor | None,
) -> torch.Tensor:
    """先 top-k 再 top-p；在**升序**排序轴上 mask，再用 scatter 还原词表顺序。"""
    if p is None and k is None:
        return logits
    vocab = logits.size(-1)
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # k<=0 表示不限制
        k = torch.where(k <= 0, vocab, k)
        # 升序下「倒数第 k」的位置即第 k 大分数的阈值
        cutoff = sorted_logits.gather(1, (vocab - k.to(torch.long)).unsqueeze(1))
        sorted_logits = sorted_logits.masked_fill(sorted_logits < cutoff, -float("inf"))

    if p is not None:
        p = torch.where(p <= 0, 1.0, p)
        probs = sorted_logits.softmax(dim=-1)
        # 累加尚未达到 1-p 的低分侧全部抹掉；强制保留最后一个（最高分）
        mask = probs.cumsum(dim=-1) <= (1 - p).unsqueeze(1)
        mask[:, -1] = False
        sorted_logits = sorted_logits.masked_fill(mask, -float("inf"))

    return torch.empty_like(sorted_logits).scatter_(1, sorted_idx, sorted_logits)


def _min_p(logits: torch.Tensor, min_p: torch.Tensor | None) -> torch.Tensor:
    """相对最高概率过滤：删掉 prob < min_p * max_prob 的位置。"""
    if min_p is None:
        return logits
    probs = torch.softmax(logits, dim=-1)
    top, _ = probs.max(dim=-1, keepdim=True)
    return logits.masked_fill(probs < min_p.unsqueeze(1) * top, -float("inf"))
