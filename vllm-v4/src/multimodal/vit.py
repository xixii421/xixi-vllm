"""Qwen3-VL Vision Transformer (patch embed + blocks + merger + DeepStack).

Attention is done with FlashInfer's ragged prefill wrapper (plan-once per
forward, run-per-block); a Triton bilinear pos-embed kernel and packed layout
keep everything else launch-light. Under ViT CUDA-graph capture we fall back to
SDPA (FlashInfer ``plan`` cannot run while a graph is capturing).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.vision import HAS_VIT_FLASHINFER, ViTAttentionBackend

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:

    @triton.jit
    def _bilinear_pos_embed_kernel(
        embed_ptr,
        output_ptr,
        H,
        W,
        h_scale,
        w_scale,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
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
            val = (
                w00_c * tl.load(embed_ptr + off00 + cols, mask=mask)
                + w01_c * tl.load(embed_ptr + off01 + cols, mask=mask)
                + w10_c * tl.load(embed_ptr + off10 + cols, mask=mask)
                + w11_c * tl.load(embed_ptr + off11 + cols, mask=mask)
            )
            tl.store(output_ptr + out_off + cols, val, mask=mask)

    def _triton_bilinear(embed_weight, t, h, w, num_grid, m_size, dtype):
        assert h % m_size == 0 and w % m_size == 0
        hidden_dim = embed_weight.shape[1]
        total_out = t * h * w
        output = torch.empty(total_out, hidden_dim, device=embed_weight.device, dtype=dtype)
        h_scale = float(num_grid - 1) / float(h - 1) if h > 1 else 0.0
        w_scale = float(num_grid - 1) / float(w - 1) if w > 1 else 0.0
        _bilinear_pos_embed_kernel[(total_out,)](
            embed_weight,
            output,
            h,
            w,
            h_scale,
            w_scale,
            num_grid,
            m_size,
            hidden_dim,
            triton.next_power_of_2(hidden_dim),
        )
        return output


def _native_bilinear(embed_weight, t, h, w, num_grid, m_size, dtype):
    assert h % m_size == 0 and w % m_size == 0
    hidden_dim = embed_weight.shape[1]
    device = embed_weight.device
    h_idxs = torch.linspace(0, num_grid - 1, h, dtype=torch.float32, device=device)
    w_idxs = torch.linspace(0, num_grid - 1, w, dtype=torch.float32, device=device)
    h_floor, w_floor = h_idxs.to(torch.long), w_idxs.to(torch.long)
    h_ceil = torch.clamp(h_floor + 1, max=num_grid - 1)
    w_ceil = torch.clamp(w_floor + 1, max=num_grid - 1)
    dh, dw = h_idxs - h_floor, w_idxs - w_floor
    dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
    h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
    h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")
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
        x = x.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.proj(x).view(-1, self.hidden_size)


class Qwen3VisionMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


def _rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def _vit_attn_sdpa(q, k, v, seqlens, offsets, scale):
    """CG-safe fallback: seqlens/offsets are Python ints (no host sync).

    q/k/v: ``[num_heads, N, head_dim]`` → out ``[num_heads, N, head_dim]``.
    """
    outputs = []
    for i, slen in enumerate(seqlens):
        if slen == 0:
            continue
        off = offsets[i]
        qi = q[:, off : off + slen].unsqueeze(0)
        ki = k[:, off : off + slen].unsqueeze(0)
        vi = v[:, off : off + slen].unsqueeze(0)
        outputs.append(F.scaled_dot_product_attention(qi, ki, vi, scale=scale).squeeze(0))
    return torch.cat(outputs, dim=1)


class Qwen3VisionAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        x,
        rotary_cos,
        rotary_sin,
        seqlens,
        offsets,
        attn_backend: Optional[ViTAttentionBackend] = None,
    ):
        if x.dim() == 3:
            x = x.squeeze(1)
        n = x.shape[0]
        qkv = self.qkv(x).view(n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each [N, H, D]
        cos = rotary_cos[:n].unsqueeze(1).float()
        sin = rotary_sin[:n].unsqueeze(1).float()
        qf, kf = q.float(), k.float()
        q = (qf * cos + _rotate_half(qf) * sin).to(q.dtype)
        k = (kf * cos + _rotate_half(kf) * sin).to(k.dtype)

        if attn_backend is not None:
            # FlashInfer ragged self-attention over the packed [N, H, D] batch.
            out = attn_backend.run(q.contiguous(), k.contiguous(), v.contiguous())
        else:
            qt, kt, vt = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
            out = _vit_attn_sdpa(qt, kt, vt, seqlens, offsets, self.scale).transpose(0, 1)

        return self.proj(out.reshape(n, -1)).unsqueeze(1)


class Qwen3VisionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Qwen3VisionAttention(hidden_size, num_heads)
        self.mlp = Qwen3VisionMLP(hidden_size, intermediate_size)

    def forward(self, x, rotary_cos, rotary_sin, seqlens, offsets, attn_backend=None):
        x = x + self.attn(
            self.norm1(x), rotary_cos, rotary_sin, seqlens, offsets, attn_backend
        )
        return x + self.mlp(self.norm2(x))


class Qwen3VisionPatchMerger(nn.Module):
    def __init__(self, hidden_size, out_hidden_size, spatial_merge_size=2, use_postshuffle_norm=False):
        super().__init__()
        self.use_postshuffle_norm = use_postshuffle_norm
        merge_hidden = hidden_size * (spatial_merge_size**2)
        norm_dim = merge_hidden if use_postshuffle_norm else hidden_size
        self.norm = nn.LayerNorm(norm_dim, eps=1e-6)
        self.fc1 = nn.Linear(merge_hidden, merge_hidden)
        self.fc2 = nn.Linear(merge_hidden, out_hidden_size)

    def forward(self, x):
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.fc1.in_features))
        else:
            x = self.norm(x).view(-1, self.fc1.in_features)
        return self.fc2(F.gelu(self.fc1(x)))


class Qwen3VisionTransformer(nn.Module):
    def __init__(self, vision_config):
        super().__init__()
        c = vision_config
        self.hidden_size = c.hidden_size
        self.num_heads = c.num_heads
        self.patch_size = getattr(c, "patch_size", 14)
        self.temporal_patch_size = getattr(c, "temporal_patch_size", 2)
        self.spatial_merge_size = getattr(c, "spatial_merge_size", 2)
        self.num_position_embeddings = getattr(c, "num_position_embeddings", 2304)
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)
        self.out_hidden_size = getattr(c, "out_hidden_size", c.hidden_size)
        self.deepstack_indexes = getattr(c, "deepstack_visual_indexes", [])
        self.depth = getattr(c, "depth", 24)

        self.patch_embed = Qwen3VisionPatchEmbed(
            getattr(c, "in_channels", 3),
            self.hidden_size,
            self.patch_size,
            self.temporal_patch_size,
        )
        self.pos_embed = nn.Embedding(self.num_position_embeddings, self.hidden_size)
        self.rotary_head_dim = self.hidden_size // self.num_heads // 2

        self.blocks = nn.ModuleList(
            [
                Qwen3VisionBlock(
                    self.hidden_size, self.num_heads, getattr(c, "intermediate_size", 4304)
                )
                for _ in range(self.depth)
            ]
        )
        self.merger = Qwen3VisionPatchMerger(
            self.hidden_size, self.out_hidden_size, self.spatial_merge_size
        )
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VisionPatchMerger(
                    self.hidden_size,
                    self.out_hidden_size,
                    self.spatial_merge_size,
                    use_postshuffle_norm=True,
                )
                for _ in self.deepstack_indexes
            ]
        )
        self._deepstack = None
        self._attn_backend: Optional[ViTAttentionBackend] = None

    @property
    def deepstack_features(self) -> Optional[list[torch.Tensor]]:
        return self._deepstack

    def _get_attn_backend(self, dtype: torch.dtype) -> Optional[ViTAttentionBackend]:
        if not HAS_VIT_FLASHINFER:
            return None
        head_dim = self.hidden_size // self.num_heads
        be = self._attn_backend
        if be is None or be.dtype != dtype:
            be = ViTAttentionBackend(
                self.num_heads, head_dim, self.pos_embed.weight.device, dtype
            )
            self._attn_backend = be
        return be

    def load_weights(self, state_dict: dict[str, torch.Tensor]):
        pfx = ""
        sd = {k: v for k, v in state_dict.items()}
        for prefix in ["model.visual.", "visual.", ""]:
            if any(k.startswith(prefix) for k in sd):
                pfx = prefix
                break
        m = dict(self.named_parameters())
        loaded = 0
        for hk, v in sd.items():
            if not hk.startswith(pfx):
                continue
            nk = hk[len(pfx) :]
            nk = nk.replace("linear_fc1", "fc1").replace("linear_fc2", "fc2")
            for hf, (fused, _shard) in {
                "attn.q.": ("attn.qkv.", "q"),
                "attn.k.": ("attn.qkv.", "k"),
                "attn.v.": ("attn.qkv.", "v"),
            }.items():
                if hf in nk:
                    nk = nk.replace(hf, fused)
            if nk in m:
                m[nk].data.copy_(v.to(dtype=m[nk].dtype))
                loaded += 1
        return loaded

    @staticmethod
    def _rot_pos_ids(h, w, m_size):
        hpos = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
        hdiv, wdiv = h // m_size, w // m_size
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
        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(0, self.rotary_head_dim, 2, dtype=torch.float32, device=device)
                / self.rotary_head_dim
            )
        )
        emb = (pos_ids.unsqueeze(-1).float() * inv_freq).flatten(1)
        emb = torch.cat((emb, emb), dim=-1)
        return (
            emb.cos().to(self.pos_embed.weight.dtype),
            emb.sin().to(self.pos_embed.weight.dtype),
        )

    def _pos_embed_interp(self, grid_list):
        fn = _triton_bilinear if HAS_TRITON else _native_bilinear
        outputs = []
        for t, h, w in grid_list:
            outputs.append(
                fn(
                    self.pos_embed.weight,
                    t,
                    h,
                    w,
                    self.num_grid_per_side,
                    self.spatial_merge_size,
                    self.pos_embed.weight.dtype,
                )
            )
        return torch.cat(outputs, dim=0)

    def prepare_metadata(self, grid_thw):
        grid_list = grid_thw.tolist() if grid_thw.dim() == 2 else grid_thw
        device = self.pos_embed.weight.device
        pos_embeds = self._pos_embed_interp(grid_list)
        rotary_cos, rotary_sin = self._rot_emb(grid_list)
        grid_np = np.array(grid_list, dtype=np.int32)
        patches = grid_np[:, 1] * grid_np[:, 2]
        cu = np.concatenate(
            [np.zeros(1, dtype=np.int32), np.repeat(patches, grid_np[:, 0]).cumsum(dtype=np.int32)]
        )
        seqlens = (cu[1:] - cu[:-1]).tolist()
        offsets = cu[:-1].tolist()
        sizes = (grid_np.prod(-1) // (self.spatial_merge_size**2)).tolist()
        max_seqlen = max(seqlens) if seqlens else 0
        return {
            "pos_embeds": pos_embeds.to(device=device),
            "rotary_cos": rotary_cos.to(device=device),
            "rotary_sin": rotary_sin.to(device=device),
            "cu_seqlens": torch.from_numpy(cu).to(device=device),
            "cu_seqlens_cpu": torch.from_numpy(cu),
            "seqlens": seqlens,
            "offsets": offsets,
            "sizes": sizes,
            "max_seqlen": max_seqlen,
        }

    def forward(
        self, pixel_values, grid_thw=None, metadata=None, out_merged=None, out_deepstack=None
    ):
        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype
        if metadata is None:
            metadata = self.prepare_metadata(grid_thw)

        hidden = self.patch_embed(pixel_values.to(device=device, dtype=dtype))
        hidden = hidden + metadata["pos_embeds"].to(dtype=hidden.dtype, device=hidden.device)
        n = hidden.shape[0]
        hidden = hidden.reshape(n, 1, -1)
        seqlens = metadata["seqlens"]
        offsets = metadata["offsets"]
        cos = metadata["rotary_cos"]
        sin = metadata["rotary_sin"]

        # FlashInfer ragged attention: plan once (identical layout across blocks).
        # Skip under CUDA-graph capture (out_merged path) — plan() is illegal there.
        attn_backend = None
        capturing = torch.cuda.is_current_stream_capturing()
        if out_merged is None and not capturing:
            cu_cpu = metadata.get("cu_seqlens_cpu")
            if cu_cpu is None:
                cu_cpu = metadata["cu_seqlens"].to("cpu")
            be = self._get_attn_backend(dtype)
            if be is not None:
                be.plan(cu_cpu)
                attn_backend = be

        deepstack = []
        for layer_num, blk in enumerate(self.blocks):
            hidden = blk(hidden, cos, sin, seqlens, offsets, attn_backend)
            if layer_num in self.deepstack_indexes:
                idx = self.deepstack_indexes.index(layer_num)
                deepstack.append(self.deepstack_merger_list[idx](hidden))

        merged = self.merger(hidden)
        if out_merged is not None:
            out_merged.copy_(merged)
            if out_deepstack is not None:
                for i, feat in enumerate(deepstack):
                    out_deepstack[i].copy_(feat)
            self._deepstack = [b for b in out_deepstack] if out_deepstack is not None else None
            return (out_merged,)

        self._deepstack = deepstack if deepstack else None
        if grid_thw is not None:
            sizes = metadata.get("sizes")
            if sizes is None:
                sizes = (grid_thw.prod(-1) // (self.spatial_merge_size**2)).tolist()
            return tuple(merged.split(sizes))
        return (merged,)
