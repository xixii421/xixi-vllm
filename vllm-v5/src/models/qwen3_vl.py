from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn
from transformers import AutoConfig, Qwen3VLConfig

from ..distributed import get_pp_group
from ..distributed.pp import IntermediateTensors, PPMissingLayer, get_layer_id
from ..layers import MRotaryEmbedding, get_mrope
from ..layers.vision import Qwen3VisionTower
from ..layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from .loader import safetensors_weights_iterator
from .qwen3 import Qwen3Attention, Qwen3DecoderLayer, Qwen3MLP, Qwen3Model, make_layers
from .sharded_loader import has_sharded_checkpoint, load_sharded_model


class Qwen3VLAttention(Qwen3Attention):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        rope_theta: float = 1000000,
        rope_scaling: dict | None = None,
        layer_id: int = 0,
        rotary_emb: MRotaryEmbedding | None = None,
    ):
        s = rope_scaling or {}
        resolved_head_dim = head_dim or hidden_size // num_heads
        rotary_emb = rotary_emb or get_mrope(
            resolved_head_dim,
            resolved_head_dim,
            max_position,
            float(s.get("rope_theta", rope_theta)),
            mrope_section=tuple(s.get("mrope_section", (24, 20, 20))),
            mrope_interleaved=bool(s.get("mrope_interleaved", True)),
        )
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_position=max_position,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            qkv_bias=qkv_bias,
            rope_theta=rope_theta,
            layer_id=layer_id,
            rotary_emb=rotary_emb,
        )


class Qwen3VLDecoderLayer(Qwen3DecoderLayer):
    def __init__(
        self,
        config,
        layer_id: int = 0,
        rotary_emb: MRotaryEmbedding | None = None,
    ):
        nn.Module.__init__(self)
        from ..layers import RMSNorm

        rope_scaling = getattr(config, "rope_scaling", None) or getattr(
            config, "rope_parameters", None
        )
        self.self_attn = Qwen3VLAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=rope_scaling if isinstance(rope_scaling, dict) else None,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            layer_id=layer_id,
            rotary_emb=rotary_emb,
        )
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class Qwen3VLModel(Qwen3Model):
    def __init__(self, config):
        nn.Module.__init__(self)
        from ..layers import RMSNorm

        self.config = config
        self.vocab_size = config.vocab_size
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size
            )
        else:
            self.embed_tokens = PPMissingLayer()
        rope_scaling = getattr(config, "rope_scaling", None) or getattr(
            config, "rope_parameters", None
        )
        s = rope_scaling if isinstance(rope_scaling, dict) else {}
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self.rotary_emb = get_mrope(
            head_dim,
            head_dim,
            config.max_position_embeddings,
            float(s.get("rope_theta", getattr(config, "rope_theta", 1000000))),
            mrope_section=tuple(s.get("mrope_section", (24, 20, 20))),
            mrope_interleaved=bool(s.get("mrope_interleaved", True)),
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda _gid, local_id: Qwen3VLDecoderLayer(
                config, layer_id=local_id, rotary_emb=self.rotary_emb
            ),
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()


class Qwen3VLForCausalLM(nn.Module):
    """Language model + vision tower weights.

    Encode/merge orchestration is injected by ``ModelRunner`` as ``mm_encoder``
    (a ``runtime.vision.VisionEncoderRunner``) so ``models`` never imports ``runtime``.
    """

    def __init__(self, config: Qwen3VLConfig, *, mm_encoder_tp_mode: str = "data"):
        super().__init__()
        self.config = config
        self.mm_encoder_tp_mode = mm_encoder_tp_mode
        self._tcfg = config.text_config
        self._ndps = len(
            getattr(config.vision_config, "deepstack_visual_indexes", []) or []
        )
        self._tie = bool(getattr(self._tcfg, "tie_word_embeddings", False))
        self.mm_encoder = None  # set by ModelRunner on first PP rank
        if get_pp_group().is_first_rank:
            self.vision_tower = Qwen3VisionTower(
                config.vision_config,
                use_weight_tp=(mm_encoder_tp_mode == "weights"),
            )
        else:
            self.vision_tower = None
        self.vl_model = Qwen3VLModel(self._tcfg)
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(self._tcfg.vocab_size, self._tcfg.hidden_size)
            if get_pp_group().is_first_rank and self._tie:
                self.lm_head.weight = self.vl_model.embed_tokens.weight
        else:
            self.lm_head = PPMissingLayer()

    def forward(
        self,
        input_ids=None,
        positions=None,
        inputs_embeds=None,
        deepstack_input_embeds=None,
        intermediate_tensors=None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.vl_model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            deepstack_input_embeds=deepstack_input_embeds,
            intermediate_tensors=intermediate_tensors,
        )

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)

    def encode_and_merge(
        self, input_ids: torch.Tensor, seqs: list, device: torch.device
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]], torch.Tensor]:
        assert get_pp_group().is_first_rank, "encode_and_merge only on first PP rank"
        assert self.mm_encoder is not None, "ModelRunner must attach mm_encoder"
        return self.mm_encoder.encode_and_merge(
            self.vl_model.embed_tokens, input_ids, seqs, device
        )

    def vit_forward_cg(self, pv, grid, metadata, out_merged, out_deepstack) -> None:
        assert self.vision_tower is not None
        self.vision_tower(
            pv,
            grid_thw=grid,
            metadata=metadata,
            out_merged=out_merged,
            out_deepstack=out_deepstack if out_deepstack else None,
        )

    def get_vit_cg_config(self):
        from ..config.vit_cg import ViTCGConfig

        vc = self.config.vision_config
        patch = getattr(vc, "patch_size", 16)
        temporal = getattr(vc, "temporal_patch_size", 2)
        in_ch = getattr(vc, "in_channels", 3)
        return ViTCGConfig(
            min_budget=64,
            max_budget=256,
            out_hidden_size=int(
                getattr(vc, "out_hidden_size", self._tcfg.hidden_size)
            ),
            merge_size=int(getattr(vc, "spatial_merge_size", 2)),
            patch_dim=in_ch * temporal * patch * patch,
            base_grid=(1, 16, 16),
            num_deepstack=self._ndps,
        )

    def load_weights(
        self,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype,
        *,
        rank: int = 0,
        tp_size: int = 1,
        pp_size: int = 1,
    ) -> None:
        from .weight_utils import (
            STACKED_GATE_UP,
            STACKED_QKV,
            assert_weights_loaded,
            finish_tied_embeddings,
            load_plain_or_tied,
            load_stacked_weight,
            move_module,
        )

        if has_sharded_checkpoint(
            model_path, rank=rank, tp_size=tp_size, pp_size=pp_size
        ):
            load_sharded_model(
                self, model_path, rank=rank, tp_size=tp_size, pp_size=pp_size
            )
            move_module(self, device=device, dtype=dtype)
            if self._tie and get_pp_group().is_first_rank and get_pp_group().is_last_rank:
                self.lm_head.weight = self.vl_model.embed_tokens.weight
            return

        visual_sd: dict[str, torch.Tensor] = {}
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        stacked = (*STACKED_QKV, *STACKED_GATE_UP)
        start_layer = self.vl_model.start_layer
        end_layer = self.vl_model.end_layer
        for name, w in safetensors_weights_iterator(model_path):
            if name.startswith("model.visual."):
                if self.vision_tower is not None:
                    visual_sd[name] = w
                continue
            our = name
            if our.startswith("model.language_model."):
                our = "vl_model." + our[len("model.language_model.") :]
            elif our.startswith("lm_head."):
                our = our
            else:
                continue
            layer_id = get_layer_id(our)
            if layer_id is not None and (
                layer_id < start_layer or layer_id >= end_layer
            ):
                continue
            if load_stacked_weight(
                name=our,
                loaded_weight=w,
                params_dict=params,
                stacked_mapping=stacked,
                loaded=loaded,
            ):
                continue
            load_plain_or_tied(
                name=our,
                loaded_weight=w,
                params_dict=params,
                loaded=loaded,
                tie_word_embeddings=self._tie,
                embed_name="vl_model.embed_tokens.weight",
            )
        if self.vision_tower is not None and visual_sd:
            vc = AutoConfig.from_pretrained(
                model_path, trust_remote_code=True
            ).vision_config
            self.vision_tower.load_weights(visual_sd, vc)
            visual_sd.clear()
        finish_tied_embeddings(
            params,
            loaded,
            tie_word_embeddings=self._tie,
            embed_name="vl_model.embed_tokens.weight",
        )
        lm_params = {
            k: v for k, v in params.items() if not k.startswith("vision_tower.")
        }
        assert_weights_loaded(lm_params, loaded, context="Qwen3VLForCausalLM")
        move_module(self, device=device, dtype=dtype)
        if self._tie and get_pp_group().is_first_rank and get_pp_group().is_last_rank:
            self.lm_head.weight = self.vl_model.embed_tokens.weight
