from __future__ import annotations

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import Optional
import triton, triton.language as tl

HAS_TRITON = True

@triton.jit
def _bilinear_pos_embed_kernel(embed_ptr, output_ptr, H, W, h_scale, w_scale, NUM_GRID: tl.constexpr, M_SIZE: tl.constexpr, HIDDEN_DIM: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    total_spatial = H * W
    spatial_idx = pid % total_spatial
    num_blocks_w = W // M_SIZE
    block_idx = spatial_idx // (M_SIZE * M_SIZE)
    local_idx = spatial_idx % (M_SIZE * M_SIZE)
    br = block_idx // num_blocks_w
    bc = block_idx % num_blocks_w
    lr = local_idx // M_SIZE
    lc = local_idx % M_SIZE
    row = br * M_SIZE + lr
    col = bc * M_SIZE + lc
    h_frac = row.to(tl.float32) * h_scale
    w_frac = col.to(tl.float32) * w_scale
    hf = tl.math.floor(h_frac).to(tl.int32)
    wf = tl.math.floor(w_frac).to(tl.int32)
    hc = tl.minimum(hf + 1, NUM_GRID - 1)
    wc = tl.minimum(wf + 1, NUM_GRID - 1)
    dh = h_frac - hf.to(tl.float32)
    dw = w_frac - wf.to(tl.float32)
    w11 = dh * dw
    w10 = dh - w11
    w01 = dw - w11
    w00 = 1.0 - dh - w01
    off00 = (hf * NUM_GRID + wf) * HIDDEN_DIM
    off01 = (hf * NUM_GRID + wc) * HIDDEN_DIM
    off10 = (hc * NUM_GRID + wf) * HIDDEN_DIM
    off11 = (hc * NUM_GRID + wc) * HIDDEN_DIM
    out_off = pid * HIDDEN_DIM
    odt = output_ptr.dtype.element_ty
    w00_c = w00.to(odt)
    w01_c = w01.to(odt)
    w10_c = w10.to(odt)
    w11_c = w11.to(odt)
    for d in tl.range(0, HIDDEN_DIM, BLOCK_D):
        cols = d + tl.arange(0, BLOCK_D)
        mask = cols < HIDDEN_DIM
        val = w00_c * tl.load(embed_ptr + off00 + cols, mask=mask) + w01_c * tl.load(embed_ptr + off01 + cols, mask=mask) + w10_c * tl.load(embed_ptr + off10 + cols, mask=mask) + w11_c * tl.load(embed_ptr + off11 + cols, mask=mask)
        tl.store(output_ptr + out_off + cols, val, mask=mask)

def _triton_bilinear(embed_weight, t, h, w, num_grid, m_size, dtype):
    assert h % m_size == 0 and w % m_size == 0
    hidden_dim = embed_weight.shape[1]
    total_out = t * h * w
    output = torch.empty(total_out, hidden_dim, device=embed_weight.device, dtype=dtype)
    h_scale = float(num_grid - 1) / float(h - 1) if h > 1 else 0.0
    w_scale = float(num_grid - 1) / float(w - 1) if w > 1 else 0.0
    _bilinear_pos_embed_kernel[total_out,](embed_weight, output, h, w, h_scale, w_scale, num_grid, m_size, hidden_dim, triton.next_power_of_2(hidden_dim))
    return output

def _native_bilinear(embed_weight, t, h, w, num_grid, m_size, dtype):
    assert h % m_size == 0 and w % m_size == 0
    hidden_dim = embed_weight.shape[1]
    device = embed_weight.device
    h_idxs = torch.linspace(0, num_grid - 1, h, dtype=torch.float32, device=device)
    w_idxs = torch.linspace(0, num_grid - 1, w, dtype=torch.float32, device=device)
    h_floor, w_floor = (h_idxs.to(torch.long), w_idxs.to(torch.long))
    h_ceil = torch.clamp(h_floor + 1, max=num_grid - 1)
    w_ceil = torch.clamp(w_floor + 1, max=num_grid - 1)
    dh, dw = (h_idxs - h_floor, w_idxs - w_floor)
    dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing='ij')
    h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing='ij')
    h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing='ij')
    w11 = dh_grid * dw_grid
    w10 = dh_grid - w11
    w01 = dw_grid - w11
    w00 = 1 - dh_grid - w01
    h_g = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
    w_g = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
    indices = (h_g * num_grid + w_g).reshape(4, -1)
    weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1).to(dtype=dtype)
    combined = (embed_weight[indices] * weights).sum(dim=0)
    combined = combined.reshape(h // m_size, m_size, w // m_size, m_size, hidden_dim)
    combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
    return combined.expand(t, -1, -1).reshape(-1, hidden_dim).to(dtype=dtype)

class Qwen3VisionPatchEmbed(nn.Module):

    def __init__(self, in_channels=3, hidden_size=1152, patch_size=14, temporal_patch_size=2):
        super().__init__()
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        k = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels, hidden_size, kernel_size=k, stride=k)

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(x).view(-1, self.hidden_size)

class Qwen3VisionMLP(nn.Module):

    def __init__(self, hidden_size, intermediate_size, *, use_weight_tp: bool = False):
        super().__init__()
        if use_weight_tp:
            from ..linear import ColumnParallelLinear, RowParallelLinear

            self.fc1 = ColumnParallelLinear(
                hidden_size, intermediate_size, bias=True, gather_output=False
            )
            self.fc2 = RowParallelLinear(intermediate_size, hidden_size, bias=True)
        else:
            self.fc1 = nn.Linear(hidden_size, intermediate_size)
            self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate='tanh'))

class Qwen3VisionAttention(nn.Module):

    def __init__(self, hidden_size, num_heads, *, use_weight_tp: bool = False):
        super().__init__()
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.use_weight_tp = use_weight_tp
        if use_weight_tp:
            from ...distributed import divide, get_tp_group
            from ..linear import QKVParallelLinear, RowParallelLinear

            tp = get_tp_group().size
            self.num_heads = divide(num_heads, tp)
            self.qkv = QKVParallelLinear(
                hidden_size, self.head_dim, num_heads, num_heads, bias=True
            )
            self.proj = RowParallelLinear(hidden_size, hidden_size, bias=True)
        else:
            self.num_heads = num_heads
            self.qkv = nn.Linear(hidden_size, hidden_size * 3)
            self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, cu_seqlens, rotary_cos, rotary_sin, max_seqlen, seqlens, offsets):
        if x.dim() == 3:
            x = x.squeeze(1)
        N = x.shape[0]
        qkv = self.qkv(x).view(N, 3, self.num_heads, self.head_dim)
        q, k, v = (qkv[:, 0], qkv[:, 1], qkv[:, 2])
        cos = rotary_cos[:N].to(dtype=q.dtype)
        sin = rotary_sin[:N].to(dtype=q.dtype)
        q_rot, k_rot = (q.float(), k.float())
        cos = cos.unsqueeze(1).float()
        sin = sin.unsqueeze(1).float()
        q = (q_rot * cos + _rotate_half(q_rot) * sin).to(q.dtype)
        k = (k_rot * cos + _rotate_half(k_rot) * sin).to(k.dtype)
        q, k, v = (q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        out = _vit_attn_sdpa(q, k, v, seqlens, offsets, self.scale)
        return self.proj(out.transpose(0, 1).reshape(N, -1)).unsqueeze(1)

def _rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)

def _vit_attn_sdpa(q, k, v, seqlens, offsets, scale):
    outputs = []
    for i, slen in enumerate(seqlens):
        if slen == 0:
            continue
        off = offsets[i]
        qi = q[:, off:off + slen].unsqueeze(0)
        ki = k[:, off:off + slen].unsqueeze(0)
        vi = v[:, off:off + slen].unsqueeze(0)
        outputs.append(F.scaled_dot_product_attention(qi, ki, vi, scale=scale).squeeze(0))
    return torch.cat(outputs, dim=1)

class Qwen3VisionBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, intermediate_size, *, use_weight_tp: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-06)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-06)
        self.attn = Qwen3VisionAttention(hidden_size, num_heads, use_weight_tp=use_weight_tp)
        self.mlp = Qwen3VisionMLP(hidden_size, intermediate_size, use_weight_tp=use_weight_tp)

    def forward(self, x, cu_seqlens, rotary_cos, rotary_sin, max_seqlen, seqlens, offsets):
        x = x + self.attn(self.norm1(x), cu_seqlens, rotary_cos, rotary_sin, max_seqlen, seqlens, offsets)
        return x + self.mlp(self.norm2(x))

class Qwen3VisionPatchMerger(nn.Module):

    def __init__(
        self,
        hidden_size,
        out_hidden_size,
        spatial_merge_size=2,
        use_postshuffle_norm=False,
        *,
        use_weight_tp: bool = False,
    ):
        super().__init__()
        self.use_postshuffle_norm = use_postshuffle_norm
        self.merge_hidden = hidden_size * spatial_merge_size ** 2
        norm_dim = self.merge_hidden if use_postshuffle_norm else hidden_size
        self.norm = nn.LayerNorm(norm_dim, eps=1e-06)
        if use_weight_tp:
            from ..linear import ColumnParallelLinear, RowParallelLinear

            self.fc1 = ColumnParallelLinear(
                self.merge_hidden, self.merge_hidden, bias=True, gather_output=False
            )
            self.fc2 = RowParallelLinear(self.merge_hidden, out_hidden_size, bias=True)
        else:
            self.fc1 = nn.Linear(self.merge_hidden, self.merge_hidden)
            self.fc2 = nn.Linear(self.merge_hidden, out_hidden_size)

    def forward(self, x):
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.merge_hidden))
        else:
            x = self.norm(x).view(-1, self.merge_hidden)
        return self.fc2(F.gelu(self.fc1(x)))

class Qwen3VisionTransformer(nn.Module):

    def __init__(self, vision_config, *, use_weight_tp: bool = False):
        super().__init__()
        c = vision_config
        self.use_weight_tp = use_weight_tp
        self.hidden_size = c.hidden_size
        self.num_heads = c.num_heads
        self.patch_size = getattr(c, 'patch_size', 14)
        self.temporal_patch_size = getattr(c, 'temporal_patch_size', 2)
        self.spatial_merge_size = getattr(c, 'spatial_merge_size', 2)
        self.num_position_embeddings = getattr(c, 'num_position_embeddings', 2304)
        self.num_grid_per_side = int(self.num_position_embeddings ** 0.5)
        self.out_hidden_size = getattr(c, 'out_hidden_size', c.hidden_size)
        self.deepstack_indexes = getattr(c, 'deepstack_visual_indexes', [])
        self.depth = getattr(c, 'depth', 24)
        self.patch_embed = Qwen3VisionPatchEmbed(getattr(c, 'in_channels', 3), self.hidden_size, self.patch_size, self.temporal_patch_size)
        self.pos_embed = nn.Embedding(self.num_position_embeddings, self.hidden_size)
        self.rotary_head_dim = self.hidden_size // self.num_heads // 2
        inter = getattr(c, 'intermediate_size', 4304)
        self.blocks = nn.ModuleList([
            Qwen3VisionBlock(
                self.hidden_size, self.num_heads, inter, use_weight_tp=use_weight_tp
            )
            for _ in range(self.depth)
        ])
        self.merger = Qwen3VisionPatchMerger(
            self.hidden_size,
            self.out_hidden_size,
            self.spatial_merge_size,
            use_weight_tp=use_weight_tp,
        )
        self.deepstack_merger_list = nn.ModuleList([
            Qwen3VisionPatchMerger(
                self.hidden_size,
                self.out_hidden_size,
                self.spatial_merge_size,
                use_postshuffle_norm=True,
                use_weight_tp=use_weight_tp,
            )
            for _ in self.deepstack_indexes
        ])
        self._deepstack = None

    @property
    def deepstack_features(self) -> Optional[list[torch.Tensor]]:
        return self._deepstack

    def load_weights(self, state_dict: dict[str, torch.Tensor]):
        pfx = ""
        sd = dict(state_dict)
        for prefix in ["model.visual.", "visual.", ""]:
            if any(k.startswith(prefix) for k in sd):
                pfx = prefix
                break
        params = dict(self.named_parameters())
        loaded = 0
        # Accumulate sharded q/k/v → fused qkv (same pattern as LM stacked load).
        qkv_buf: dict[str, dict[str, torch.Tensor]] = {}
        shard_map = {
            "attn.q.": ("attn.qkv.", "q"),
            "attn.k.": ("attn.qkv.", "k"),
            "attn.v.": ("attn.qkv.", "v"),
        }

        def _load_plain(param: nn.Parameter, tensor: torch.Tensor) -> None:
            loader = getattr(param, "weight_loader", None)
            if loader is not None:
                loader(param, tensor)
            else:
                param.data.copy_(tensor)

        for hk, v in sd.items():
            if not hk.startswith(pfx):
                continue
            nk = hk[len(pfx) :].replace("linear_fc1", "fc1").replace("linear_fc2", "fc2")
            sharded = False
            for hf, (fused, shard) in shard_map.items():
                if hf in nk:
                    fused_name = nk.replace(hf, fused)
                    qkv_buf.setdefault(fused_name, {})[shard] = v
                    sharded = True
                    break
            if sharded:
                continue
            if nk in params:
                param = params[nk]
                _load_plain(param, v.to(dtype=param.dtype))
                loaded += 1
        for fused_name, shards in qkv_buf.items():
            if fused_name not in params or not {"q", "k", "v"} <= shards.keys():
                continue
            dest = params[fused_name]
            loader = getattr(dest, "weight_loader", None)
            if loader is not None and self.use_weight_tp:
                for shard_id in ("q", "k", "v"):
                    loader(dest, shards[shard_id].to(dtype=dest.dtype), shard_id)
            else:
                q, k, v = shards["q"], shards["k"], shards["v"]
                dest.data.copy_(torch.cat([q, k, v], dim=0).to(dtype=dest.dtype))
            loaded += 1
        return loaded

    @staticmethod
    def _rot_pos_ids(h, w, m_size):
        hpos = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
        hdiv, wdiv = (h // m_size, w // m_size)
        hpos = hpos.reshape(hdiv, m_size, wdiv, m_size).transpose(0, 2, 1, 3).flatten()
        wpos = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
        wpos = wpos.reshape(hdiv, m_size, wdiv, m_size).transpose(0, 2, 1, 3).flatten()
        return torch.from_numpy(np.stack([hpos, wpos], axis=-1))

    def _rot_emb(self, grid_list):
        pos_ids = []
        for t, h, w in grid_list:
            ids = self._rot_pos_ids(h, w, self.spatial_merge_size)
            if t > 1:
                ids = ids.repeat(t, 1)
            pos_ids.append(ids)
        pos_ids = torch.cat(pos_ids, dim=0)
        device = self.pos_embed.weight.device
        pos_ids = pos_ids.to(device)
        inv_freq = 1.0 / 10000 ** (torch.arange(0, self.rotary_head_dim, 2, dtype=torch.float32, device=device) / self.rotary_head_dim)
        emb = (pos_ids.unsqueeze(-1).float() * inv_freq).flatten(1)
        emb = torch.cat((emb, emb), dim=-1)
        return (emb.cos().to(self.pos_embed.weight.dtype), emb.sin().to(self.pos_embed.weight.dtype))

    def _pos_embed_interp(self, grid_list):
        fn = _triton_bilinear if HAS_TRITON else _native_bilinear
        outputs = []
        for t, h, w in grid_list:
            outputs.append(fn(self.pos_embed.weight, t, h, w, self.num_grid_per_side, self.spatial_merge_size, self.pos_embed.weight.dtype))
        return torch.cat(outputs, dim=0)

    def prepare_metadata(self, grid_thw):
        grid_list = grid_thw.tolist() if grid_thw.dim() == 2 else grid_thw
        device = self.pos_embed.weight.device
        pos_embeds = self._pos_embed_interp(grid_list)
        rotary_cos, rotary_sin = self._rot_emb(grid_list)
        grid_np = np.array(grid_list, dtype=np.int32)
        patches = grid_np[:, 1] * grid_np[:, 2]
        cu = np.concatenate([np.zeros(1, dtype=np.int32), np.repeat(patches, grid_np[:, 0]).cumsum(dtype=np.int32)])
        seqlens = (cu[1:] - cu[:-1]).tolist()
        offsets = cu[:-1].tolist()
        sizes = (grid_np.prod(-1) // self.spatial_merge_size ** 2).tolist()
        max_seqlen = max(seqlens) if seqlens else 0
        return {'pos_embeds': pos_embeds.to(device=device), 'rotary_cos': rotary_cos.to(device=device), 'rotary_sin': rotary_sin.to(device=device), 'cu_seqlens': torch.from_numpy(cu).to(device=device), 'seqlens': seqlens, 'offsets': offsets, 'sizes': sizes, 'max_seqlen': max_seqlen}

    def forward(self, pixel_values, grid_thw=None, metadata=None, out_merged=None, out_deepstack=None):
        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype
        if metadata is None:
            metadata = self.prepare_metadata(grid_thw)
        hidden = self.patch_embed(pixel_values.to(device=device, dtype=dtype))
        hidden = hidden + metadata['pos_embeds'].to(dtype=hidden.dtype, device=hidden.device)
        N = hidden.shape[0]
        hidden = hidden.reshape(N, 1, -1)
        seqlens = metadata['seqlens']
        offsets = metadata['offsets']
        deepstack = []
        for layer_num, blk in enumerate(self.blocks):
            hidden = blk(hidden, metadata['cu_seqlens'], metadata['rotary_cos'], metadata['rotary_sin'], metadata['max_seqlen'], seqlens, offsets)
            if layer_num in self.deepstack_indexes:
                idx = self.deepstack_indexes.index(layer_num)
                deepstack.append(self.deepstack_merger_list[idx](hidden))
        merged = self.merger(hidden)
        dps = deepstack if deepstack else None
        self._deepstack = dps
        if out_merged is not None:
            # CUDA-graph path: write into captured buffers.
            out_merged.copy_(merged)
            if out_deepstack is not None and dps is not None:
                for i, feat in enumerate(dps):
                    out_deepstack[i].copy_(feat)
            return out_merged, (
                list(out_deepstack) if out_deepstack is not None else dps
            )
        return merged, dps
