"""词表并行：嵌入按 vocab 维切分，输出前 all-reduce；LM head 先切分计算再 all-gather。"""

from __future__ import annotations

import torch
from torch import nn

from ...distributed import (
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)


def _pad_vocab_size(vocab_size: int, tp_size: int) -> int:
    return ((vocab_size + tp_size - 1) // tp_size) * tp_size


class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        tp = get_tp_group()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.vocab_start_index, self.vocab_end_index = self._get_vocab_range()
        self.weight = nn.Parameter(
            torch.empty(self.vocab_end_index - self.vocab_start_index, embedding_dim)
        )
        setattr(self.weight, "weight_loader", self.weight_loader)

    def _get_vocab_range(self) -> tuple[int, int]:
        tp = get_tp_group()
        padded = _pad_vocab_size(self.num_embeddings, tp.size)
        per_partition = padded // tp.size
        return per_partition * tp.group_rank, per_partition * (tp.group_rank + 1)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int = 0
    ) -> None:
        tp = get_tp_group()
        padded = _pad_vocab_size(self.num_embeddings, tp.size)
        per_partition = padded // tp.size
        if loaded_weight.shape[0] < padded:
            # 原权重不足整块：补齐到 tp 整数倍再切分。
            padded_weight = torch.empty(
                padded, loaded_weight.shape[1], dtype=loaded_weight.dtype
            )
            padded_weight[: loaded_weight.shape[0]].copy_(loaded_weight)
            loaded_weight = padded_weight
        start = per_partition * tp.group_rank
        param.data.copy_(loaded_weight[start : start + per_partition])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 超出本 rank 词表范围的 id 必须产出 0，all-reduce 汇总后才是唯一正确行。
        # 用 mask 置零而非 clamp：all_reduce 是求和，非持有 rank 若产出非零行会污染结果。
        mask = (x < self.vocab_start_index) | (x >= self.vocab_end_index)
        shifted = (x - self.vocab_start_index).masked_fill(mask, 0)
        out = torch.nn.functional.embedding(shifted, self.weight)
        out = out.masked_fill(mask.unsqueeze(-1), 0.0)
        return tensor_model_parallel_all_reduce(out)


class ParallelLMHead(nn.Module):
    """输出维（vocab）切分的 LM head：各 rank 算局部 logits 后 all-gather。"""

    def __init__(self, hidden_size: int, vocab_size: int, *, bias: bool = False):
        super().__init__()
        tp = get_tp_group()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.padded_vocab_size = _pad_vocab_size(vocab_size, tp.size)
        self.output_size_per_partition = self.padded_vocab_size // tp.size
        self.vocab_start_index = self.output_size_per_partition * tp.group_rank
        self.vocab_end_index = self.output_size_per_partition * (tp.group_rank + 1)
        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, hidden_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
        else:
            self.register_parameter("bias", None)
        setattr(self.weight, "weight_loader", self.weight_loader)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int = 0
    ) -> None:
        tp = get_tp_group()
        per_partition = self.padded_vocab_size // tp.size
        if loaded_weight.shape[0] < self.padded_vocab_size:
            padded_weight = torch.empty(
                self.padded_vocab_size, loaded_weight.shape[1], dtype=loaded_weight.dtype
            )
            padded_weight[: loaded_weight.shape[0]].copy_(loaded_weight)
            loaded_weight = padded_weight
        start = per_partition * tp.group_rank
        param.data.copy_(loaded_weight[start : start + per_partition])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.nn.functional.linear(hidden_states, self.weight, self.bias)
        logits = tensor_model_parallel_all_gather(logits, dim=-1)
        return logits[..., : self.vocab_size]
