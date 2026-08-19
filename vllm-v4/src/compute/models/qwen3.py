from __future__ import annotations

import torch
from torch import nn
from transformers.models.qwen3 import Qwen3Config

from ..layers import (
    Attention,
    MergedColumnParallelLinear,
    ParallelLMHead,
    QKVParallelLinear,
    RMSNorm,
    RotaryEmbedding,
    RowParallelLinear,
    SiluAndMul,
    VocabParallelEmbedding,
)
from ..layers.attention import AttnInputs
from ..layers.linear import get_weight_loader
from ...distributed import (
    IntermediateTensors,
    PPMissingLayer,
    get_pp_group,
    get_pp_indices,
)


def _rope_theta(config: Qwen3Config) -> float:
    """读取 RoPE base：优先 ``rope_parameters``（transformers >= 5.x 的新字段），
    兼容旧版顶层 ``rope_theta`` 字段，兜底 Qwen3 官方默认 1e6。
    """
    rp = getattr(config, "rope_parameters", None)
    if isinstance(rp, dict) and rp.get("rope_theta") is not None:
        return float(rp["rope_theta"])
    return float(getattr(config, "rope_theta", 1_000_000))


def _parse_layer_idx(name: str) -> int | None:
    """从 HF 权重名解析全局层下标，如 ``model.layers.15.self_attn...`` -> 15。"""
    prefix = "model.layers."
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    idx = rest.split(".", 1)[0]
    try:
        return int(idx)
    except ValueError:
        return None


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # gate/up 融合为单个 [2*intermediate, hidden] 阵，TP 下按输出维切分。
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size], bias=False
        )
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
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
        self.head_dim = head_dim or hidden_size // num_heads
        # QKVParallelLinear 内部按 tp 切分 head，暴露 per-rank 头数。
        self.qkv_proj = QKVParallelLinear(
            hidden_size, self.head_dim, num_heads, num_kv_heads, bias=qkv_bias
        )
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # attention 输出已按 head 切分（每 rank 只算自己 head），输入维不再二次切分。
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim, hidden_size, input_is_parallel=True
        )
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, self.head_dim, max_position, rope_theta
        )
        # layer_id 为本 PP stage 内的本地层下标（KV 池按 stage 分片后以本地下标索引）。
        self.attn = Attention(
            self.num_heads, self.head_dim, self.num_kv_heads, layer_id=layer_id
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
        pp = get_pp_group()
        self.start_layer, self.end_layer = get_pp_indices(
            config.num_hidden_layers, pp.group_rank, pp.size
        )
        # 非本 stage 的组件用 PPMissingLayer 占位，保持命名空间与参数名稳定。
        self.embed_tokens = (
            VocabParallelEmbedding(config.vocab_size, config.hidden_size)
            if pp.is_first_rank
            else PPMissingLayer()
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, layer_id=local_i)
                for local_i in range(self.end_layer - self.start_layer)
            ]
        )
        self.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if pp.is_last_rank
            else PPMissingLayer()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn: AttnInputs,
        intermediate_tensors: IntermediateTensors | None = None,
    ):
        if intermediate_tensors is None:
            h, residual = self.embed_tokens(input_ids), None
        else:
            h = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in self.layers:
            h, residual = layer(positions, h, residual, attn)
        if get_pp_group().is_last_rank:
            return self.norm(h, residual)[0]
        return IntermediateTensors(hidden_states=h, residual=residual)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = (
            ParallelLMHead(config.hidden_size, config.vocab_size, bias=False)
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )

    def forward(self, input_ids, positions, attn: AttnInputs, intermediate_tensors=None):
        return self.model(input_ids, positions, attn, intermediate_tensors)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)

    def argmax(self, hidden_states):
        """贪心快速路径：直接取全局 argmax token id，省去全量 logits 的 all-gather。"""
        return self.lm_head.argmax(hidden_states)

    def load_weights(self, weights):
        # 融合阵（qkv / gate_up）在加载时按 head 或输出维切分写入。
        stacked = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params = dict(self.named_parameters())
        loaded = set()
        skipped = []

        for name, w in weights:
            # PP：跳过非本 stage 的层，并把全局层下标重映射为本地下标。
            layer_idx = _parse_layer_idx(name)
            if layer_idx is not None:
                if not (self.model.start_layer <= layer_idx < self.model.end_layer):
                    continue
                name = name.replace(
                    f"model.layers.{layer_idx}.",
                    f"model.layers.{layer_idx - self.model.start_layer}.",
                    1,
                )

            for param_name, weight_name, shard_id in stacked:
                if weight_name not in name:
                    continue
                our = name.replace(weight_name, param_name)
                if our in params:
                    params[our].weight_loader(params[our], w, shard_id)
                    loaded.add(our)
                else:
                    skipped.append(name)
                break
            else:
                if name in params:
                    get_weight_loader(params[name])(params[name], w)
                    loaded.add(name)
                else:
                    skipped.append(name)

        # 权重绑定：lm_head 复用 embed_tokens（仅 pp=1 时两者同 rank；Qwen3 实际
        # 已在 checkpoint 中显式保存 lm_head.weight，此分支作为兜底）。
        # 注意 embed 权重多一行"越界占位行"（VocabParallelEmbedding 的 +1 设计），
        # 绑定到 lm_head 时需切片跳过，否则 shape 不匹配。
        unloaded = set(params) - loaded
        if "lm_head.weight" in unloaded and "model.embed_tokens.weight" in loaded:
            embed = params["model.embed_tokens.weight"].data
            params["lm_head.weight"].data.copy_(embed[1:])
            loaded.add("lm_head.weight")
            unloaded.discard("lm_head.weight")

        if unloaded:
            print(f"[WARN] {len(unloaded)} params NOT loaded: {sorted(unloaded)[:10]}...")
        if skipped:
            print(f"[INFO] {len(skipped)} HF weights skipped (not in model): {skipped[:5]}...")
