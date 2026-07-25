import glob
import os

import torch
from torch import nn
from transformers import AutoConfig, Qwen3VLConfig

from ..layers import get_mrope
from ..multimodal import EncoderCache, Qwen3VisionTower, mrope_positions
from .loader import default_weight_loader, safetensors_weights_iterator
from .qwen3 import Qwen3Attention, Qwen3DecoderLayer, Qwen3Model


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
        rope_theta: float = 1_000_000,
        rope_scaling: dict | None = None,
        layer_id: int = 0,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_position=min(max_position, 8192),
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            qkv_bias=qkv_bias,
            rope_theta=rope_theta,
            layer_id=layer_id,
        )
        s = rope_scaling or {}
        self.rotary_emb = get_mrope(
            self.head_dim,
            self.head_dim,
            max_position,
            float(s.get("rope_theta", rope_theta)),
            mrope_section=tuple(s.get("mrope_section", (24, 20, 20))),
            mrope_interleaved=bool(s.get("mrope_interleaved", True)),
        )


class Qwen3VLDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_id: int = 0):
        nn.Module.__init__(self)
        from ..layers import RMSNorm
        from .qwen3 import Qwen3MLP

        scaling = getattr(config, "rope_scaling", None) or getattr(config, "rope_parameters", None)
        self.self_attn = Qwen3VLAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rope_theta=getattr(config, "rope_theta", 1_000_000),
            rope_scaling=scaling if isinstance(scaling, dict) else None,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            layer_id=layer_id,
        )
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3VLModel(Qwen3Model):
    def __init__(self, config):
        nn.Module.__init__(self)
        from ..layers import RMSNorm

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3VLDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        rope = self.layers[0].self_attn.rotary_emb
        for layer in self.layers[1:]:
            layer.self_attn.rotary_emb = rope
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3VLForCausalLM(nn.Module):
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self._tcfg = config.text_config
        self.vision_tower = Qwen3VisionTower(config.vision_config)
        self.vl_model = Qwen3VLModel(self._tcfg)
        self.lm_head = nn.Linear(self._tcfg.hidden_size, self._tcfg.vocab_size, bias=False)
        self._tie = bool(getattr(self._tcfg, "tie_word_embeddings", False))
        self._ndps = len(getattr(config.vision_config, "deepstack_visual_indexes", []) or [])
        self._encoder_cache = EncoderCache(256)

    def forward(self, input_ids=None, positions=None, inputs_embeds=None, deepstack_input_embeds=None):
        return self.vl_model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            deepstack_input_embeds=deepstack_input_embeds,
        )

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)

    def _mrope_module(self):
        return self.vl_model.layers[0].self_attn.rotary_emb

    def load_weights(self, model_path: str, device: torch.device, dtype: torch.dtype) -> None:
        hf_sd = {}
        for f in sorted(glob.glob(os.path.join(model_path, "model*.safetensors"))):
            from safetensors import safe_open

            with safe_open(f, framework="pt", device="cpu") as s:
                hf_sd.update({k: s.get_tensor(k) for k in s.keys() if k.startswith("model.visual.")})
        vc = AutoConfig.from_pretrained(model_path, trust_remote_code=True).vision_config
        self.vision_tower.load_weights(hf_sd, vc)

        params = dict(self.named_parameters())
        attn0 = self.vl_model.layers[0].self_attn
        q = attn0.num_heads * attn0.head_dim
        kv = attn0.num_kv_heads * attn0.head_dim
        gate_up: dict[str, list] = {}

        for name, w in safetensors_weights_iterator(model_path):
            if name.startswith("model.visual."):
                continue
            if name.startswith("model.language_model."):
                our = "vl_model." + name[len("model.language_model.") :]
            elif name.startswith("lm_head."):
                our = name
            else:
                continue

            fused = None
            for hf, offset, size in (
                ("q_proj", 0, q),
                ("k_proj", q, kv),
                ("v_proj", q + kv, kv),
            ):
                if f".{hf}." in our or our.endswith(f".{hf}.weight") or our.endswith(f".{hf}.bias"):
                    our2 = our.replace(hf, "qkv_proj")
                    if our2 in params:
                        params[our2].data[offset : offset + size].copy_(w)
                    fused = True
                    break
            if fused:
                continue
            if "gate_proj" in our:
                our2 = our.replace("gate_proj", "gate_up_proj")
                gate_up.setdefault(our2, [None, None])[0] = w
            elif "up_proj" in our:
                our2 = our.replace("up_proj", "gate_up_proj")
                gate_up.setdefault(our2, [None, None])[1] = w
            elif our in params:
                default_weight_loader(params[our], w)

        for our2, (g, u) in gate_up.items():
            if g is not None and u is not None and our2 in params:
                default_weight_loader(params[our2], torch.cat([g, u], 0))

        if self._tie and "vl_model.embed_tokens.weight" in params:
            self.lm_head.weight.data = self.vl_model.embed_tokens.weight.data
        for p in self.parameters():
            p.data = p.data.to(device=device, dtype=dtype)
        for name, buf in self.named_buffers():
            if buf is None:
                continue
            parent, _, leaf = name.rpartition(".")
            mod = self.get_submodule(parent) if parent else self
            setattr(mod, leaf, buf.to(device=device, dtype=dtype if buf.is_floating_point() else buf.dtype))
        if self._tie:
            self.lm_head.weight.data = self.vl_model.embed_tokens.weight.data

    def encode_and_merge(self, input_ids, seqs, device):
        """Prefill: ViT encode (window + cache) → scatter embeds → MRoPE positions."""
        text = self.vl_model.embed_tokens(input_ids)
        vis = self._encode_visual(seqs, device)
        embeds = text.clone()
        off = vi = 0
        for seq in seqs:
            cs, ce = seq.num_cached_tokens, seq.num_cached_tokens + seq.num_scheduled_tokens
            for mf in seq.mm_features:
                a, b = mf.start, mf.start + mf.num_encoder_tokens
                s, e = max(a, cs), min(b, ce)
                if s < e:
                    local = off + (s - cs)
                    embeds[local : local + (e - s)] = vis[vi][s - a : e - a].to(
                        dtype=embeds.dtype, device=device
                    )
                vi += 1
            off += seq.num_scheduled_tokens
        positions = mrope_positions(input_ids, seqs, self.vision_tower.merge_size, device)
        self._mrope_module().ensure_cache_size(int(getattr(positions, "max_pos", 0)) + 1)
        return embeds, self._build_deepstack(seqs, embeds.shape[0], embeds.shape[1], device, embeds.dtype), positions

    def _encode_visual(self, seqs, device):
        """Encode images overlapping the scheduled window; hit EncoderCache when possible."""
        entries = [
            (mf, seq.num_cached_tokens, seq.num_cached_tokens + seq.num_scheduled_tokens)
            for seq in seqs
            for mf in seq.mm_features
        ]
        out: list[torch.Tensor | None] = [None] * len(entries)
        miss_i, miss_mf = [], []
        for i, (mf, cs, ce) in enumerate(entries):
            if max(mf.start, cs) >= min(mf.start + mf.num_encoder_tokens, ce):
                continue
            hit = self._encoder_cache.get(mf.uid)
            if hit is not None:
                emb, ds = hit
                out[i] = emb.to(device=device, non_blocking=True)
                if ds is not None:
                    mf.deepstack = [t.to(device=device, non_blocking=True) for t in ds]
                continue
            miss_i.append(i)
            miss_mf.append(mf)

        if not miss_mf:
            return out

        m = self.vision_tower.merge_size
        pv = torch.cat([mf.data for mf in miss_mf], 0).to(device)
        grids, sizes = [], []
        for mf in miss_mf:
            if mf.grid_thw is None:
                grids.append(torch.tensor([[1, m * 2, m * 2]], device=device))
                sizes.append(mf.num_encoder_tokens)
            else:
                t, h, w = (int(x) for x in mf.grid_thw.detach().cpu().view(-1, 3)[0].tolist())
                grids.append(torch.tensor([[t, h, w]], device=device, dtype=torch.int64))
                sizes.append((t * h * w) // (m * m))
        grid = torch.cat(grids, 0)
        result = self.vision_tower(pv, grid_thw=grid)
        flat = torch.cat(list(result), 0) if isinstance(result, tuple) else result
        dps = self.vision_tower.deepstack_features

        cur = 0
        for idx, mf, sz in zip(miss_i, miss_mf, sizes):
            emb = flat[cur : cur + sz]
            ds = [t[cur : cur + sz].contiguous() for t in dps] if dps is not None else None
            if ds is not None:
                mf.deepstack = ds
            out[idx] = emb
            self._encoder_cache.put(mf.uid, emb, ds)
            cur += sz
        return out

    def _build_deepstack(self, seqs, n, h, device, dtype):
        if not self._ndps:
            return None
        if not any(getattr(mf, "deepstack", None) for seq in seqs for mf in seq.mm_features):
            return None
        bufs = [torch.zeros(n, h, device=device, dtype=dtype) for _ in range(self._ndps)]
        off = 0
        for seq in seqs:
            cs, ce = seq.num_cached_tokens, seq.num_cached_tokens + seq.num_scheduled_tokens
            for mf in seq.mm_features:
                cached = getattr(mf, "deepstack", None)
                if cached is None:
                    continue
                a, b = mf.start, mf.start + mf.num_encoder_tokens
                s, e = max(a, cs), min(b, ce)
                if s >= e:
                    continue
                local = off + (s - cs)
                for li in range(self._ndps):
                    bufs[li][local : local + (e - s)].copy_(
                        cached[li][s - a : e - a].to(device=device, dtype=dtype)
                    )
            off += seq.num_scheduled_tokens
        return bufs
