import torch
import torch.nn.functional as F
from torch import nn
from ..distributed import get_tp_group, tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce
from ..distributed.math import divide
DEFAULT_VOCAB_PADDING_SIZE = 64

def pad_vocab_size(vocab_size: int, pad_to: int=DEFAULT_VOCAB_PADDING_SIZE) -> int:
    return (vocab_size + pad_to - 1) // pad_to * pad_to

class VocabParallelEmbedding(torch.nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, params_dtype: torch.dtype | None=None, padding_size: int=DEFAULT_VOCAB_PADDING_SIZE):
        super().__init__()
        self.tp_rank = get_tp_group().group_rank
        self.tp_size = get_tp_group().size
        self.num_embeddings = num_embeddings
        self.padding_size = padding_size
        padded = pad_vocab_size(self.num_embeddings, self.padding_size)
        if padded % self.tp_size != 0:
            padded = pad_vocab_size(padded, self.tp_size)
        self.num_embeddings_padded = padded
        self.embedding_dim = embedding_dim
        self.num_embeddings_per_partition = divide(self.num_embeddings_padded, self.tp_size)
        self.vocab_start_index = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, self.embedding_dim, dtype=params_dtype))
        setattr(self.weight, 'weight_loader', self.weight_loader)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        start_idx = self.vocab_start_index
        shard_size = self.num_embeddings_per_partition
        assert loaded_weight.shape[0] == self.num_embeddings, f'self.num_embeddings={self.num_embeddings!r} loaded_weight.shape[0]={loaded_weight.shape[0]!r}'
        if start_idx >= self.num_embeddings:
            param.data.fill_(0)
            return
        copy_len = min(shard_size, self.num_embeddings - start_idx)
        loaded = loaded_weight.narrow(0, start_idx, copy_len)
        param[:copy_len].data.copy_(loaded)
        if copy_len < shard_size:
            param[copy_len:].data.fill_(0)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            input_mask = (x >= self.vocab_start_index) & (x < self.vocab_end_index)
            x = x - self.vocab_start_index
            x.masked_fill_(~input_mask, 0)
            output_parallel = F.embedding(x.long(), self.weight)
            output_parallel.masked_fill_(~input_mask.unsqueeze(-1), 0)
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = F.embedding(x, self.weight)
        return output

class ParallelLMHead(VocabParallelEmbedding):

    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool=False, params_dtype: torch.dtype | None=None, padding_size: int=DEFAULT_VOCAB_PADDING_SIZE):
        super().__init__(num_embeddings, embedding_dim, params_dtype=params_dtype, padding_size=padding_size)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition, dtype=params_dtype))
            setattr(self.bias, 'weight_loader', self.weight_loader)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor):
        logits = F.linear(x, self.weight, self.bias)
        logits = tensor_model_parallel_all_gather(logits)
        logits = logits[:, :self.num_embeddings]
        return logits
