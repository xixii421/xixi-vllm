"""Unit: vocab shard padding loader + DeepStack PP config gate."""

from __future__ import annotations

import pytest
import torch

from src.config import EngineConfig
from src.distributed.pp import get_pp_indices
from src.layers.vocab_parallel_embedding import VocabParallelEmbedding, pad_vocab_size
from tests.helpers import VL_MODEL, require_model


def test_pad_vocab_size():
    assert pad_vocab_size(1000, 64) == 1024
    assert pad_vocab_size(1024, 64) == 1024


def test_get_pp_indices_remainder_on_last():
    assert get_pp_indices(28, 0, 2) == (0, 14)
    assert get_pp_indices(28, 1, 2) == (14, 28)
    assert get_pp_indices(5, 0, 2) == (0, 2)
    assert get_pp_indices(5, 1, 2) == (2, 5)


def test_vocab_weight_loader_handles_padding(monkeypatch):
    """Last TP shard may extend past real vocab; loader must zero-pad."""
    from src.distributed import parallel_state as ps

    class _FakeGroup:
        size = 2
        group_rank = 1

    monkeypatch.setattr(ps, "TP", _FakeGroup())
    monkeypatch.setattr(ps, "get_tp_group", lambda: _FakeGroup())

    vocab = 100
    emb = VocabParallelEmbedding(vocab, 8)
    assert emb.num_embeddings_padded == 128
    assert emb.num_embeddings_per_partition == 64
    assert emb.vocab_start_index == 64

    full = torch.arange(vocab * 8, dtype=torch.float32).reshape(vocab, 8)
    emb.weight_loader(emb.weight, full)
    assert torch.equal(emb.weight[:36], full[64:100])
    assert torch.all(emb.weight[36:] == 0)


@require_model(VL_MODEL)
def test_deepstack_pp_config_rejects_too_shallow_first_stage(monkeypatch):
    import src.distributed.pp as du

    orig = du.get_pp_indices

    def _tiny_first(num_hidden_layers, pp_rank, pp_size):
        if pp_rank == 0:
            return 0, 1  # owns only layer 0; DeepStack needs 0..2
        return orig(num_hidden_layers, pp_rank, pp_size)

    monkeypatch.setattr(du, "get_pp_indices", _tiny_first)
    with pytest.raises(ValueError, match="DeepStack"):
        EngineConfig(
            model=str(VL_MODEL),
            tp_size=1,
            pp_size=2,
            enforce_eager=True,
            num_kvcache_blocks=8,
        )
