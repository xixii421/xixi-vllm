from __future__ import annotations
from collections.abc import Iterable

import torch
from torch import nn
from transformers.models.qwen3 import Qwen3Config

from .layers.activation import SiluAndMul
from .layers.attention import Attention
from .layers.layernorm import RMSNorm
from .layers.rotary import RotaryEmbedding
from .model_loader import default_weight_loader


# HF 的q/k/v投影权重分成三段, 分别对应q, k, v的投影权重,按照位移切片写入
# 内存布局:[q | k | v]
_QKV_SHARDS = (
    ("q_proj", lambda q, kv: (0, q)),
    ("k_proj", lambda q, kv: (q, kv)),
    ("v_proj", lambda q, kv: (q + kv, kv)),
)


class Qwen3MLP(nn.Module):
    """SwiGLU 前馈网络"""
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        # gate决定放行多少, up提供候选值, down回到hidden
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiluAndMul 约定后半是 gate：silu(gate) * up
        return self.down_proj(
            self.act_fn(torch.cat([self.up_proj(x), self.gate_proj(x)], dim=-1))
        )


class Qwen3Attention(nn.Module):
    """Qwen3 自注意力: FusedQKV -> QK Norm -> RoPE -> GQA -> Linear"""
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        # 小模型可能显式给 head_dim
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # 一次线性同时产出 Q|K|V，加载时再按切片写入三段
        self.qkv_proj = nn.Linear(
            config.hidden_size, self.q_size + 2 * self.kv_size, bias=False
        )
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)
        # RoPE 按照绝对位置旋转Q, K; Qwen3 全维旋转 (rotary_dim == head_dim)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            self.head_dim,
            config.max_position_embeddings,
            float(getattr(config, "rope_theta", 1_000_000)),
        )
        self.attn = Attention(self.num_heads, self.head_dim, self.num_kv_heads)
        # QK-Norm：按每个 head 的 head_dim 做 RMSNorm
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        # 一次投影 -> 三段切分 -> 分别做 norm 和 RoPE -> 重新组装 -> 线性投影
        q, k, v = self.qkv_proj(hidden_states).split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        # [B,S,H*D] → [B,S,H,D] 逐 head 做 norm，再拍平回 [B,S,H*D]；
        # V 不做 norm 和 RoPE
        q = self.q_norm(q.unflatten(-1, (-1, self.head_dim))).flatten(-2)
        k = self.k_norm(k.unflatten(-1, (-1, self.head_dim))).flatten(-2)
        q, k = self.rotary_emb(positions, q, k)

        # 显式分出 head 维：Q 有 num_heads 头，K/V 只有 num_kv_heads 头（GQA）
        b, s, _ = q.shape
        q = q.view(b, s, self.num_heads, self.head_dim)
        k = k.view(b, s, self.num_kv_heads, self.head_dim)
        v = v.view(b, s, self.num_kv_heads, self.head_dim)
        # Attention 内部完成 GQA 广播与缩放点积注意力，输出压平后过 o_proj
        return self.o_proj(self.attn(q, k, v).flatten(-2))


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 解码器层: 自注意 -> 残差 -> MLP -> 残差"""
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        # 输入层 norm 与残差融合；输出层 norm 与残差融合
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 首层：residual 取 embedding；之后各层在 RMSNorm 内完成 h+residual 再归一化
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        # Attn 输出再归一化,然后过 MLP
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class Qwen3Model(nn.Module):
    """Qwen3 模型: Embed -> 堆叠Layers -> Norm"""
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # residual 初始值为None,首层Layer直接用 embedding 作为残差
        h, residual = self.embed_tokens(input_ids), None
        for layer in self.layers:
            h, residual = layer(positions, h, residual)
        # final norm 同时完成最后一次残差融合；只返回归一化后的表征
        return self.norm(h, residual)[0]


class Qwen3ForCausalLM(nn.Module):
    """Qwen3模型顶层: backbone + LM head"""
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # 只算Backbone,推理引擎需要hidden state时再算LM head
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 表征计算与词表投影分离,后续张量并行时lm_head可以单独切分
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        attn0 = self.model.layers[0].self_attn
        q_size, kv_size = attn0.q_size, attn0.kv_size

        for name, w in weights:
            merged = False
            for hf, off_fn in _QKV_SHARDS:
                if f".{hf}.weight" not in name:
                    continue
                fused = name.replace(hf, "qkv_proj")    # HF 名 -> 本项目名
                param = params[fused]
                off, size = off_fn(q_size, kv_size)
                # 切片写入，不是整段覆盖
                param.data[off : off + size].copy_(w)
                loaded.add(fused)
                merged = True
                break
            if merged:
                continue        # QKV已经处理,跳到下一个权重
            if name in params:
                default_weight_loader(params[name], w)
                loaded.add(name)

        if "lm_head.weight" not in loaded and "model.embed_tokens.weight" in loaded:
            default_weight_loader(
                params["lm_head.weight"], params["model.embed_tokens.weight"]
            )
            loaded.add("lm_head.weight")
        # 检查是否有未加载的权重
        missing = set(params) - loaded
        if missing:
            print(f"[WARN] {len(missing)} params not loaded: {sorted(missing)[:8]}...")
