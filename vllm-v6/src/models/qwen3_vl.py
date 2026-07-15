from __future__ import annotations
import glob
import hashlib
import os
from typing import Optional, Union
import torch
from torch import nn
from transformers import AutoConfig, Qwen3VLConfig
from ..distributed import get_pp_group
from ..layers import get_mrope
from ..distributed.pp import IntermediateTensors, PPMissingLayer, get_layer_id
from ..layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from ..multimodal import EncoderCache, Qwen3VisionTower, mrope_positions
from .loader import default_weight_loader, safetensors_weights_iterator
from .qwen3 import Qwen3Attention, Qwen3DecoderLayer, Qwen3MLP, Qwen3Model, make_layers

class Qwen3VLAttention(Qwen3Attention):

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, max_position: int=4096 * 32, head_dim: int | None=None, rms_norm_eps: float=1e-06, qkv_bias: bool=False, rope_theta: float=1000000, rope_scaling: dict | None=None, layer_id: int=0):
        super().__init__(hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads, max_position=min(max_position, 8192), head_dim=head_dim, rms_norm_eps=rms_norm_eps, qkv_bias=qkv_bias, rope_theta=rope_theta, layer_id=layer_id)
        s = rope_scaling or {}
        self.rotary_emb = get_mrope(self.head_dim, self.head_dim, max_position, float(s.get('rope_theta', rope_theta)), mrope_section=tuple(s.get('mrope_section', (24, 20, 20))), mrope_interleaved=bool(s.get('mrope_interleaved', True)))

class Qwen3VLDecoderLayer(Qwen3DecoderLayer):

    def __init__(self, config, layer_id: int=0):
        nn.Module.__init__(self)
        from ..layers import RMSNorm
        rope_scaling = getattr(config, 'rope_scaling', None) or getattr(config, 'rope_parameters', None)
        self.self_attn = Qwen3VLAttention(hidden_size=config.hidden_size, num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads, max_position=config.max_position_embeddings, rope_theta=getattr(config, 'rope_theta', 1000000), rope_scaling=rope_scaling if isinstance(rope_scaling, dict) else None, rms_norm_eps=config.rms_norm_eps, qkv_bias=getattr(config, 'attention_bias', False), head_dim=getattr(config, 'head_dim', None), layer_id=layer_id)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

class Qwen3VLModel(Qwen3Model):

    def __init__(self, config):
        nn.Module.__init__(self)
        from ..layers import RMSNorm
        self.config = config
        self.vocab_size = config.vocab_size
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(config.num_hidden_layers, lambda _gid, local_id: Qwen3VLDecoderLayer(config, layer_id=local_id))
        real = [self.layers[i] for i in range(self.start_layer, self.end_layer)]
        if real:
            shared_rope = real[0].self_attn.rotary_emb
            for layer in real[1:]:
                layer.self_attn.rotary_emb = shared_rope
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

class Qwen3VLForCausalLM(nn.Module):

    def __init__(self, config: Qwen3VLConfig, *, mm_encoder_tp_mode: str='data'):
        super().__init__()
        self.config = config
        self.mm_encoder_tp_mode = mm_encoder_tp_mode
        self._tcfg = config.text_config
        self._ndps = len(getattr(config.vision_config, 'deepstack_visual_indexes', []) or [])
        self._encoder_cache = EncoderCache(256)
        self._vit_cg = None
        self._tie = bool(getattr(self._tcfg, 'tie_word_embeddings', False))
        if get_pp_group().is_first_rank:
            self.vision_tower = Qwen3VisionTower(config.vision_config, disable_tp=True)
        else:
            self.vision_tower = None
        self.vl_model = Qwen3VLModel(self._tcfg)
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(self._tcfg.vocab_size, self._tcfg.hidden_size)
            if get_pp_group().is_first_rank and self._tie:
                self.lm_head.weight = self.vl_model.embed_tokens.weight
        else:
            self.lm_head = PPMissingLayer()

    def forward(self, input_ids=None, positions=None, inputs_embeds=None, deepstack_input_embeds=None, intermediate_tensors=None) -> Union[torch.Tensor, IntermediateTensors]:
        return self.vl_model(input_ids=input_ids, positions=positions, inputs_embeds=inputs_embeds, deepstack_input_embeds=deepstack_input_embeds, intermediate_tensors=intermediate_tensors)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)

    def load_weights(self, model_path: str, device: torch.device, dtype: torch.dtype) -> None:
        if self.vision_tower is not None:
            hf_sd: dict[str, torch.Tensor] = {}
            for f in sorted(glob.glob(os.path.join(model_path, 'model*.safetensors'))):
                from safetensors import safe_open
                with safe_open(f, framework='pt', device='cpu') as s:
                    hf_sd.update({k: s.get_tensor(k) for k in s.keys() if k.startswith('model.visual.')})
            vc = AutoConfig.from_pretrained(model_path, trust_remote_code=True).vision_config
            self.vision_tower.load_weights(hf_sd, vc)
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        stacked = [('qkv_proj', 'q_proj', 'q'), ('qkv_proj', 'k_proj', 'k'), ('qkv_proj', 'v_proj', 'v'), ('gate_up_proj', 'gate_proj', 0), ('gate_up_proj', 'up_proj', 1)]
        start_layer = self.vl_model.start_layer
        end_layer = self.vl_model.end_layer
        for name, w in safetensors_weights_iterator(model_path):
            if name.startswith('model.visual.'):
                continue
            our = name
            if our.startswith('model.language_model.'):
                our = 'vl_model.' + our[len('model.language_model.'):]
            elif our.startswith('lm_head.'):
                our = our
            else:
                continue
            layer_id = get_layer_id(our)
            if layer_id is not None and (layer_id < start_layer or layer_id >= end_layer):
                continue
            for param_name, weight_name, shard_id in stacked:
                if weight_name not in our:
                    continue
                our2 = our.replace(weight_name, param_name)
                if our2 not in params:
                    break
                param = params[our2]
                weight_loader = getattr(param, 'weight_loader', None)
                if weight_loader is not None:
                    weight_loader(param, w, shard_id)
                else:
                    default_weight_loader(param, w)
                loaded.add(our2)
                break
            else:
                if our == 'vl_model.embed_tokens.weight' and 'lm_head.weight' in params and ('vl_model.embed_tokens.weight' not in params):
                    param = params['lm_head.weight']
                    weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                    weight_loader(param, w)
                    loaded.add('lm_head.weight')
                    continue
                if our in params:
                    param = params[our]
                    weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                    weight_loader(param, w)
                    loaded.add(our)
        if 'lm_head.weight' in params and 'lm_head.weight' not in loaded and ('vl_model.embed_tokens.weight' in params):
            default_weight_loader(params['lm_head.weight'], params['vl_model.embed_tokens.weight'])
        for p in self.parameters():
            p.data = p.data.to(device=device, dtype=dtype)
        for name, buf in self.named_buffers():
            if buf is not None:
                parent, _, leaf = name.rpartition('.')
                mod = self.get_submodule(parent) if parent else self
                setattr(mod, leaf, buf.to(device=device, dtype=dtype if buf.is_floating_point() else buf.dtype))
        if self._tie and get_pp_group().is_first_rank and get_pp_group().is_last_rank:
            self.lm_head.weight = self.vl_model.embed_tokens.weight

    def encode_and_merge(self, input_ids: torch.Tensor, seqs: list, device: torch.device) -> tuple[torch.Tensor, Optional[list[torch.Tensor]], torch.Tensor]:
        assert get_pp_group().is_first_rank, 'encode_and_merge only on first PP rank'
        assert self.vision_tower is not None
        text_emb = self.vl_model.embed_tokens(input_ids)
        all_vis = self._encode_visual(seqs, device)
        n, h = text_emb.shape
        embeds = text_emb.clone()
        off = vis_i = 0
        for seq in seqs:
            chunk_start = seq.num_cached_tokens
            chunk_end = chunk_start + seq.num_scheduled_tokens
            for mf in seq.mm_features:
                img_start = mf.start
                img_end = mf.start + mf.num_encoder_tokens
                ov_start = max(img_start, chunk_start)
                ov_end = min(img_end, chunk_end)
                if ov_start >= ov_end:
                    vis_i += 1
                    continue
                vis_off = ov_start - img_start
                nv = ov_end - ov_start
                local = off + (ov_start - chunk_start)
                embeds[local:local + nv] = all_vis[vis_i][vis_off:vis_off + nv].to(dtype=embeds.dtype, device=device)
                vis_i += 1
            off += seq.num_scheduled_tokens
        dps = self._build_deepstack(seqs, n, h, device, embeds.dtype)
        positions = mrope_positions(input_ids, seqs, self.vision_tower.merge_size, device)
        return (embeds, dps, positions)

    def _encode_visual(self, seqs: list, device: torch.device) -> list[torch.Tensor]:
        assert self.vision_tower is not None
        from ..distributed import get_tp_group
        from ..multimodal.vision_dp import run_dp_sharded_vision_encode
        mfs = [mf for seq in seqs for mf in seq.mm_features]
        out: list[torch.Tensor | None] = [None] * len(mfs)
        miss_idx: list[int] = []
        miss_mfs: list = []
        use_dp = self.mm_encoder_tp_mode == 'data' and get_tp_group().size > 1
        if use_dp:
            miss_idx = list(range(len(mfs)))
            miss_mfs = list(mfs)
        else:
            for i, mf in enumerate(mfs):
                cached = self._encoder_cache.get(mf.uid)
                if cached is not None:
                    emb, ds = cached
                    out[i] = emb.to(device=device, non_blocking=True)
                    if ds is not None:
                        mf.deepstack = [t.to(device=device, non_blocking=True) for t in ds]
                    continue
                miss_idx.append(i)
                miss_mfs.append(mf)
        if miss_mfs:
            m = self.vision_tower.merge_size
            pv = torch.cat([mf.data for mf in miss_mfs], dim=0).to(device)
            grids = []
            sizes = []
            for mf in miss_mfs:
                g = mf.grid_thw
                if g is None:
                    grids.append(torch.tensor([[1, m * 2, m * 2]], device=device))
                    sizes.append(mf.num_encoder_tokens)
                else:
                    g = g.to(device).view(-1, 3)
                    grids.append(g)
                    sizes.append(int(g.prod(-1).item() // (m * m)))
            grid = torch.cat(grids, dim=0)
            dps_batch: list[torch.Tensor] | None = None
            cg = self._vit_cg
            used_cg = False
            if not use_dp and cg is not None and cg.can_use(grid):
                try:
                    flat, dps_batch = cg.replay(pv, grid)
                    if dps_batch is not None and self.vision_tower.vit is not None:
                        self.vision_tower.vit._deepstack = dps_batch
                    used_cg = True
                except Exception:
                    used_cg = False
            if not used_cg:
                if use_dp:
                    flat, dps_batch = run_dp_sharded_vision_encode(self.vision_tower, pv, grid)
                    if dps_batch is not None and self.vision_tower.vit is not None:
                        self.vision_tower.vit._deepstack = dps_batch
                else:
                    result = self.vision_tower(pv, grid_thw=grid)
                    flat = torch.cat(list(result), dim=0) if isinstance(result, tuple) else result
                    dps_batch = self.vision_tower.deepstack_features
            cursor = 0
            for idx, mf, sz in zip(miss_idx, miss_mfs, sizes):
                emb = flat[cursor:cursor + sz]
                out[idx] = emb
                ds_i = None
                if dps_batch is not None:
                    ds_i = [t[cursor:cursor + sz].contiguous() for t in dps_batch]
                    mf.deepstack = ds_i
                if not use_dp:
                    self._encoder_cache.put(mf.uid, emb, ds_i)
                cursor += sz
        return out

    def vit_forward_cg(self, pv, grid, metadata, out_merged, out_deepstack) -> None:
        assert self.vision_tower is not None
        self.vision_tower(pv, grid_thw=grid, metadata=metadata, out_merged=out_merged, out_deepstack=out_deepstack if out_deepstack else None)

    def get_vit_cg_config(self):
        from ..cuda_graph.vit import ViTCGConfig
        vc = self.config.vision_config
        patch = getattr(vc, 'patch_size', 16)
        temporal = getattr(vc, 'temporal_patch_size', 2)
        in_ch = getattr(vc, 'in_channels', 3)
        return ViTCGConfig(min_budget=64, max_budget=256, out_hidden_size=int(getattr(vc, 'out_hidden_size', self._tcfg.hidden_size)), merge_size=int(getattr(vc, 'spatial_merge_size', 2)), patch_dim=in_ch * temporal * patch * patch, base_grid=(1, 16, 16), num_deepstack=self._ndps)

    def _build_deepstack(self, seqs, n, h, device, dtype):
        if not self._ndps:
            return None
        if not any((getattr(mf, 'deepstack', None) is not None for seq in seqs for mf in seq.mm_features)):
            return None
        bufs = [torch.zeros(n, h, device=device, dtype=dtype) for _ in range(self._ndps)]
        off = 0
        for seq in seqs:
            chunk_start = seq.num_cached_tokens
            chunk_end = chunk_start + seq.num_scheduled_tokens
            for mf in seq.mm_features:
                cached = getattr(mf, 'deepstack', None)
                if cached is None:
                    continue
                img_start = mf.start
                img_end = mf.start + mf.num_encoder_tokens
                ov_start = max(img_start, chunk_start)
                ov_end = min(img_end, chunk_end)
                if ov_start >= ov_end:
                    continue
                vis_off = ov_start - img_start
                nv = ov_end - ov_start
                local = off + (ov_start - chunk_start)
                for li in range(self._ndps):
                    bufs[li][local:local + nv].copy_(cached[li][vis_off:vis_off + nv].to(device=device, dtype=dtype))
            off += seq.num_scheduled_tokens
        return bufs

def make_mm_uid(pixel_values: torch.Tensor) -> str:
    raw = pixel_values.detach().float().cpu().numpy().tobytes()
    return hashlib.sha1(raw[:4096] + str(tuple(pixel_values.shape)).encode()).hexdigest()
