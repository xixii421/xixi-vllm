from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ..distributed import (
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from ..distributed.math import divide

DEFAULT_VOCAB_PADDING_SIZE = 64


def pad_vocab_size(
    vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE
) -> int:
    return (vocab_size + pad_to - 1) // pad_to * pad_to


class VocabParallelEmbedding(nn.Module):
    """Shard vocabulary rows across the tensor-parallel group."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
    ) -> None:
        super().__init__()
        tp = get_tp_group()
        self.tp_rank = tp.group_rank
        self.tp_size = tp.size
        self.num_embeddings = num_embeddings

        padded = pad_vocab_size(num_embeddings, padding_size)
        if padded % self.tp_size:
            padded = pad_vocab_size(padded, self.tp_size)
        self.num_embeddings_padded = padded
        self.embedding_dim = embedding_dim
        self.num_embeddings_per_partition = divide(padded, self.tp_size)
        self.vocab_start_index = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition

        self.weight = nn.Parameter(
            torch.empty(
                self.num_embeddings_per_partition,
                embedding_dim,
                dtype=params_dtype,
            )
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        assert loaded_weight.shape[0] == self.num_embeddings
        start = self.vocab_start_index
        size = self.num_embeddings_per_partition
        if start >= self.num_embeddings:
            param.data.zero_()
            return

        length = min(size, self.num_embeddings - start)
        param.data[:length].copy_(loaded_weight.narrow(0, start, length))
        if length < size:
            param.data[length:].zero_()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return F.embedding(token_ids, self.weight)

        is_local = (token_ids >= self.vocab_start_index) & (
            token_ids < self.vocab_end_index
        )
        local_ids = token_ids - self.vocab_start_index
        local_ids.masked_fill_(~is_local, 0)
        output = F.embedding(local_ids.long(), self.weight)
        output.masked_fill_(~is_local.unsqueeze(-1), 0)
        return tensor_model_parallel_all_reduce(output)


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.num_embeddings_per_partition, dtype=params_dtype)
            )
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Local vocab projection: ``[tokens, local_vocab]``."""
        return F.linear(x, self.weight, self.bias)

    def gather_logits(self, local_logits: torch.Tensor) -> torch.Tensor:
        """All-gather shard logits into a full vocabulary (minus padding)."""
        if self.tp_size == 1:
            return local_logits[..., : self.num_embeddings]
        logits = tensor_model_parallel_all_gather(local_logits)
        return logits[..., : self.num_embeddings]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compatibility path: local project + full-vocab gather."""
        return self.gather_logits(self.project(x))

    def greedy_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Vocab-parallel greedy argmax without materializing full logits.

        Communication is ``O(batch * tp_size)`` instead of ``O(batch * vocab)``.
        """
        local = self.project(x)

        valid = max(0, min(self.vocab_end_index, self.num_embeddings) - self.vocab_start_index)
        if valid < local.shape[-1]:
            local = local.clone()
            local[..., valid:] = float("-inf")

        local_vals, local_idx = local.max(dim=-1)
        global_idx = local_idx + self.vocab_start_index
        if self.tp_size == 1:
            return global_idx

        pair = torch.stack((local_vals.float(), global_idx.float()), dim=-1).contiguous()
        buf = [torch.empty_like(pair) for _ in range(self.tp_size)]
        dist.all_gather(buf, pair, group=get_tp_group().device_group)
        gathered = torch.stack(buf, dim=1)  # [batch, tp, 2]
        best = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        return gathered[:, :, 1].gather(dim=-1, index=best).squeeze(-1).to(torch.long)
