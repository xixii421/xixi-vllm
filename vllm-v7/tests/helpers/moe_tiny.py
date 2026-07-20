"""Build a tiny Qwen3-MoE checkpoint for fast GPU/distributed tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from src.utils.paths import resolve_hf_home


def build_tiny_moe(
    out: Path,
    *,
    num_experts: int = 8,
    num_layers: int = 2,
    num_heads: int = 4,
    head_dim: int = 64,
) -> Path:
    """FlashInfer on this host needs head_dim=64; 8 experts / I=128 fits TP=2/4."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cfg = Qwen3MoeConfig(
        hidden_size=num_heads * head_dim,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=max(2, num_heads // 2),
        intermediate_size=512,
        moe_intermediate_size=128,
        num_experts=num_experts,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        vocab_size=1024,
        max_position_embeddings=256,
        head_dim=head_dim,
        rms_norm_eps=1e-6,
        rope_theta=1000000,
        attention_bias=False,
        tie_word_embeddings=False,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        hidden_act="silu",
    )
    torch.manual_seed(0)
    Qwen3MoeForCausalLM(cfg).eval().save_pretrained(out, safe_serialization=True)
    cfg.save_pretrained(out)
    tok_src = resolve_hf_home() / "Qwen3-8B"
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ):
        p = tok_src / name
        if p.exists():
            shutil.copy(p, out / name)
    return out
