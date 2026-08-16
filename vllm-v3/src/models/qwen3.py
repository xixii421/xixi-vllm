from __future__ import annotations

import torch
from torch import nn
from transformers.models.qwen3 import Qwen3Config

from ..layers import Attention, RMSNorm, RotaryEmbedding, SiluAndMul
from ..layers.attention import AttnInputs
from .loader import default_weight_loader


def _rope_theta(config: Qwen3Config) -> float:
    """读取 RoPE base：优先 ``rope_parameters``（transformers >= 5.x 的新字段），
    兼容旧版顶层 ``rope_theta`` 字段，兜底 Qwen3 官方默认 1e6。
    """
    rp = getattr(config, "rope_parameters", None)
    if isinstance(rp, dict) and rp.get("rope_theta") is not None:
        return float(rp["rope_theta"])
    return float(getattr(config, "rope_theta", 1_000_000))


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # gate_proj/up_proj 融合为单个 [2*intermediate, hidden] GEMM（少一次 launch+HBM 往返）
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x = self.gate_up_proj(x)
        return self.down_proj(self.act_fn(x))


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 32768,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        rope_theta: float = 1_000_000,
        layer_id: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = nn.Linear(
            hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim, bias=qkv_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, self.head_dim, max_position, rope_theta)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.num_kv_heads,
            layer_id=layer_id,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, attn: AttnInputs):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = self._split_norm_rope(positions, qkv)
        return self.o_proj(self.attn(q, k, v, attn))

    def _split_norm_rope(self, positions, qkv):
        """qkv → split → q/k norm → rope，产出 (q, k, v) 供 attention 使用。"""
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, _ = self.q_norm(q)
        k, _ = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_id: int = 0):
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rope_theta=_rope_theta(config),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            layer_id=layer_id,
        )
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual: torch.Tensor | None, attn: AttnInputs):
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, attn)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_id=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, attn: AttnInputs):
        h, residual = self.embed_tokens(input_ids), None
        for layer in self.layers:
            h, residual = layer(positions, h, residual, attn)
        return self.norm(h, residual)[0]


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, positions, attn: AttnInputs):
        return self.model(input_ids, positions, attn)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        skipped = []

        attn = self.model.layers[0].self_attn
        _q_size = attn.q_size
        _kv_size = attn.kv_size
        # Collect gate/up weights for fusion into gate_up_proj
        gate_up_bufs: dict[str, tuple] = {}

        for name, w in weights:
            # merge q_proj/k_proj/v_proj → qkv_proj
            for hf, our, offset, size in [
                ("q_proj", "qkv_proj", 0, _q_size),
                ("k_proj", "qkv_proj", _q_size, _kv_size),
                ("v_proj", "qkv_proj", _q_size + _kv_size, _kv_size),
            ]:
                if hf in name:
                    our_name = name.replace(hf, our)
                    params[our_name].data[offset:offset + size].copy_(w)
                    loaded.add(our_name)
                    break
            else:
                # merge gate_proj + up_proj → gate_up_proj (fused MLP)
                if "gate_proj" in name:
                    our = name.replace("gate_proj", "gate_up_proj")
                    gate_up_bufs.setdefault(our, [None, None])
                    gate_up_bufs[our][0] = w
                elif "up_proj" in name:
                    our = name.replace("up_proj", "gate_up_proj")
                    gate_up_bufs.setdefault(our, [None, None])
                    gate_up_bufs[our][1] = w
                elif name in params:
                    default_weight_loader(params[name], w)
                    loaded.add(name)
                else:
                    skipped.append(name)

        for our, (g_w, u_w) in gate_up_bufs.items():
            if g_w is not None and u_w is not None and our in params:
                default_weight_loader(params[our], torch.cat([g_w, u_w], dim=0))
                loaded.add(our)

        unloaded = set(params) - loaded
        if "lm_head.weight" in unloaded and "model.embed_tokens.weight" in loaded:
            default_weight_loader(params["lm_head.weight"], params["model.embed_tokens.weight"])
            loaded.add("lm_head.weight")
            unloaded.discard("lm_head.weight")

        if unloaded:
            print(f"[WARN] {len(unloaded)} params NOT loaded: {sorted(unloaded)[:10]}...")
        if skipped:
            print(f"[INFO] {len(skipped)} HF weights skipped (not in model): {skipped[:5]}...")
