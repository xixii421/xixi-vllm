from __future__ import annotations
import torch
from torch import nn


class Sampler(nn.Module):
    """批量 token 采样器。

    支持 greedy、temperature、top-k、top-p 和 min-p 的混合批处理采样。
    当 temperature 小于等于阈值时，对应行按 greedy 处理。
    """

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor | None = None,
        top_ks: torch.Tensor | None = None,
        min_ps: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """执行批量采样。

        Args:
            logits: [batch_size, vocab_size] 未归一化 logits。
            temperatures: [batch_size] 采样温度；小于等于阈值时视为 greedy。
            top_ps: 可选，[batch_size] nucleus 采样阈值。
            top_ks: 可选，[batch_size] top-k 采样阈值。
            min_ps: 可选，[batch_size] min-p 采样阈值。
            generator: 可选随机数生成器。

        Returns:
            [batch_size] 采样得到的 token ids。
        """
        device = logits.device
        vocab = logits.size(-1)

        temperatures = temperatures.to(dtype=torch.float32, device=device)

        # temperature 过低的行视为 greedy。
        greedy_mask = temperatures <= 1e-5

        # 全批 greedy 时直接 argmax，跳过后续采样路径。
        if bool(greedy_mask.all()):
            return logits.argmax(dim=-1)

        # 归一化各采样参数，并判断对应策略是否实际启用。
        top_ks, use_top_k = self._prep_top_k(top_ks, vocab, device)
        top_ps, use_top_p = self._prep_top_p(top_ps, device)
        min_ps, use_min_p = self._prep_min_p(min_ps, device)

        # greedy 行临时使用温度 1.0，避免除零；最终结果会被 argmax 覆盖。
        safe_temp = torch.where(
            greedy_mask, torch.ones_like(temperatures), temperatures
        )
        scaled = logits.float() / safe_temp.unsqueeze(1)

        if use_top_k:
            # top-k 先将候选集压缩到当前 batch 中最大的 k，
            # 后续 top-p / min-p 过滤都在更小的有序空间中完成。
            max_k = int(top_ks.max().item())
            vals, idx = torch.topk(scaled, max_k, dim=-1)

            # 每行只保留前 k 个候选，其余位置置为 -inf。
            col = torch.arange(max_k, device=device)
            vals = vals.masked_fill(
                col.unsqueeze(0) >= top_ks.unsqueeze(1), float("-inf")
            )

            sample = self._sample_ordered(
                vals,
                idx,
                top_ps if use_top_p else None,
                min_ps if use_min_p else None,
                generator,
            )
        elif use_top_p:
            # 无 top-k 但启用 top-p 时，需要完整降序排序后再做 nucleus 过滤。
            vals, idx = torch.sort(scaled, descending=True, dim=-1)
            sample = self._sample_ordered(
                vals, idx, top_ps, min_ps if use_min_p else None, generator
            )
        elif use_min_p:
            # 仅启用 min-p：基于每行最大概率过滤低概率 token。
            probs = torch.softmax(scaled, dim=-1)
            max_p = probs.max(dim=-1, keepdim=True).values
            probs = probs.masked_fill(probs < min_ps.unsqueeze(1) * max_p, 0.0)
            sample = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        else:
            # 普通 temperature sampling。
            probs = torch.softmax(scaled, dim=-1)
            sample = torch.multinomial(probs, 1, generator=generator).squeeze(1)

        # 混合 batch：greedy 行最终用 argmax 结果覆盖采样结果。
        if bool(greedy_mask.any()):
            sample = torch.where(greedy_mask, logits.argmax(dim=-1), sample)
        return sample

    @staticmethod
    def _sample_ordered(
        vals: torch.Tensor,
        idx: torch.Tensor,
        top_ps: torch.Tensor | None,
        min_ps: torch.Tensor | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """在降序排列后的候选空间内应用 top-p / min-p，并采样返回原始 token id。

        Args:
            vals: [batch_size, candidates] 已降序排列的 logits 或分数。
            idx: [batch_size, candidates] 与 vals 对应的原始 token ids。
            top_ps: 可选，[batch_size] nucleus 采样阈值。
            min_ps: 可选，[batch_size] min-p 采样阈值。
            generator: 可选随机数生成器。

        Returns:
            [batch_size] 采样得到的原始 token ids。
        """
        probs = torch.softmax(vals, dim=-1)

        if top_ps is not None or min_ps is not None:
            remove = torch.zeros_like(probs, dtype=torch.bool)

            if top_ps is not None:
                # 当前 token 之前的累计概率已经超过阈值时，移除当前 token。
                cum = probs.cumsum(dim=-1)
                remove |= (cum - probs) > top_ps.unsqueeze(1)

            if min_ps is not None:
                # 有序空间的首列即最大概率，作为 min-p 的阈值基准。
                remove |= probs < min_ps.unsqueeze(1) * probs[:, :1]

            # 始终保留概率最高的 token，避免过滤后概率全为零。
            remove[:, 0] = False
            probs = probs.masked_fill(remove, 0.0)

        pos = torch.multinomial(probs, 1, generator=generator)
        return idx.gather(1, pos).squeeze(1)

    @staticmethod
    def _prep_top_k(
        top_ks: torch.Tensor | None, vocab: int, device: torch.device
    ) -> tuple[torch.Tensor | None, bool]:
        """归一化 top-k 参数，并返回是否实际启用 top-k。"""
        if top_ks is None:
            return None, False

        k = top_ks.to(dtype=torch.long, device=device).clamp_(max=vocab)

        # k <= 0 表示关闭 top-k，等价于保留整个词表。
        k = torch.where(k <= 0, torch.full_like(k, vocab), k)

        # 只有存在小于词表大小的 k 时，top-k 才真正生效。
        return k, bool((k < vocab).any())

    @staticmethod
    def _prep_top_p(
        top_ps: torch.Tensor | None, device: torch.device
    ) -> tuple[torch.Tensor | None, bool]:
        """归一化 top-p 参数，并返回是否实际启用 top-p。"""
        if top_ps is None:
            return None, False

        p = top_ps.to(dtype=torch.float32, device=device)

        # p <= 0 表示关闭 top-p，等价于阈值 1.0。
        p = torch.where(p <= 0, torch.ones_like(p), p)

        # 只有存在小于 1.0 的阈值时，top-p 才真正生效。
        return p, bool((p < 1.0).any())

    @staticmethod
    def _prep_min_p(
        min_ps: torch.Tensor | None, device: torch.device
    ) -> tuple[torch.Tensor | None, bool]:
        """归一化 min-p 参数，并返回是否实际启用 min-p。"""
        if min_ps is None:
            return None, False

        mp = min_ps.to(dtype=torch.float32, device=device)

        # 只有存在大于 0 的阈值时，min-p 才真正生效。
        return mp, bool((mp > 0).any())