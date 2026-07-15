"""Qwen3 causal LM with Megatron-style TP + PP (teaching mini-vLLM)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Optional, Union

import torch
from torch import nn
from transformers.models.qwen3 import Qwen3Config

from ..distributed import get_pp_group, get_pp_indices, get_tp_group
from ..layers import Attention, RMSNorm, RotaryEmbedding, SiluAndMul
from ..layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from ..layers.utils_parallel import IntermediateTensors, PPMissingLayer, get_layer_id
from ..layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from .loader import default_weight_loader


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        layer_id: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tp_group().size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, self.head_dim, max_position, rope_theta
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            self.num_kv_heads,
            layer_id=layer_id,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.rotary_emb(positions, q, k)

        out = self.attn(q, k, v)
        out, _ = self.o_proj(out.flatten(1, -1))
        return out


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_id: int = 0):
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rope_theta=getattr(config, "rope_theta", 1_000_000),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            layer_id=layer_id,
        )
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, positions, hidden_states, residual: torch.Tensor | None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


def make_layers(
    num_hidden_layers: int,
    layer_fn: Callable[[int, int], nn.Module],
) -> tuple[int, int, torch.nn.ModuleList]:
    """Build layers for this PP rank.

    ``layer_fn(global_layer_id, local_kv_layer_id)`` — local id indexes the
    per-rank KV pool (0 .. num_local_layers-1).
    """
    start_layer, end_layer = get_pp_indices(
        num_hidden_layers, get_pp_group().group_rank, get_pp_group().size
    )
    modules = torch.nn.ModuleList(
        [PPMissingLayer() for _ in range(start_layer)]
        + [
            layer_fn(idx, idx - start_layer)
            for idx in range(start_layer, end_layer)
        ]
        + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
    )
    return start_layer, end_layer, modules


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda _gid, local_id: Qwen3DecoderLayer(
                config=config, layer_id=local_id
            ),
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        deepstack_input_embeds: list[torch.Tensor] | None = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                assert input_ids is not None
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        n_ds = len(deepstack_input_embeds) if deepstack_input_embeds else 0
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, residual)
            # DeepStack (VL): HF injects features into LLM layers 0 .. n_ds-1
            # (ViT `deepstack_visual_indexes` only picks which *vision* layers extract).
            # Those early layers always live on the first PP stage when pp_size is small.
            local_i = i - self.start_layer if get_pp_group().is_first_rank else -1
            if get_pp_group().is_first_rank and local_i >= 0 and local_i < n_ds:
                hidden_states = hidden_states + deepstack_input_embeds[local_i]

        if not get_pp_group().is_last_rank:
            assert hidden_states is not None and residual is not None
            return IntermediateTensors(
                hidden_states=hidden_states, residual=residual
            )
        return self.norm(hidden_states, residual)[0]


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
            if get_pp_group().is_first_rank and getattr(
                config, "tie_word_embeddings", True
            ):
                self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self.lm_head = PPMissingLayer()

    def forward(
        self,
        input_ids=None,
        positions=None,
        inputs_embeds=None,
        deepstack_input_embeds=None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            deepstack_input_embeds=deepstack_input_embeds,
            intermediate_tensors=intermediate_tensors,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            hidden_states=torch.empty(
                batch_size, self.config.hidden_size, dtype=dtype, device=device
            ),
            residual=torch.empty(
                batch_size, self.config.hidden_size, dtype=dtype, device=device
            ),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()

        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if layer_id is not None and (
                layer_id < self.model.start_layer or layer_id >= self.model.end_layer
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    break
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                loaded.add(name)
                break
            else:
                # Tied embeddings: last PP rank needs embed weights as lm_head.
                if (
                    name == "model.embed_tokens.weight"
                    and "lm_head.weight" in params_dict
                    and "model.embed_tokens.weight" not in params_dict
                ):
                    param = params_dict["lm_head.weight"]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded.add("lm_head.weight")
                    continue

                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded.add(name)

        # Tie on last==first rank when checkpoint omitted lm_head.
        if (
            "lm_head.weight" in params_dict
            and "lm_head.weight" not in loaded
            and "model.embed_tokens.weight" in params_dict
        ):
            default_weight_loader(
                params_dict["lm_head.weight"],
                params_dict["model.embed_tokens.weight"],
            )
