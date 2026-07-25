import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ..layers.vision import ViTAttentionBackend


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
    spatial_idx = pid % (H * W)
    num_blocks_w = W // M_SIZE
    block_idx = spatial_idx // (M_SIZE * M_SIZE)
    local_idx = spatial_idx % (M_SIZE * M_SIZE)
    row = (block_idx // num_blocks_w) * M_SIZE + local_idx // M_SIZE
    col = (block_idx % num_blocks_w) * M_SIZE + local_idx % M_SIZE
    h_frac = row.to(tl.float32) * h_scale
    w_frac = col.to(tl.float32) * w_scale
    hf = tl.math.floor(h_frac).to(tl.int32)
    wf = tl.math.floor(w_frac).to(tl.int32)
    hc = tl.minimum(hf + 1, NUM_GRID - 1)
    wc = tl.minimum(wf + 1, NUM_GRID - 1)
    dh = h_frac - hf.to(tl.float32)
    dw = w_frac - wf.to(tl.float32)
    w11, w10, w01 = dh * dw, dh - dh * dw, dw - dh * dw
    w00 = 1.0 - dh - w01
    odt = output_ptr.dtype.element_ty
    for d in tl.range(0, HIDDEN_DIM, BLOCK_D):
        cols = d + tl.arange(0, BLOCK_D)
        mask = cols < HIDDEN_DIM
        val = (
            w00.to(odt) * tl.load(embed_ptr + (hf * NUM_GRID + wf) * HIDDEN_DIM + cols, mask=mask)
            + w01.to(odt) * tl.load(embed_ptr + (hf * NUM_GRID + wc) * HIDDEN_DIM + cols, mask=mask)
            + w10.to(odt) * tl.load(embed_ptr + (hc * NUM_GRID + wf) * HIDDEN_DIM + cols, mask=mask)
            + w11.to(odt) * tl.load(embed_ptr + (hc * NUM_GRID + wc) * HIDDEN_DIM + cols, mask=mask)
        )
        tl.store(output_ptr + pid * HIDDEN_DIM + cols, val, mask=mask)


def _bilinear(embed_weight, t, h, w, num_grid, m_size, dtype):
    assert h % m_size == 0 and w % m_size == 0
    hidden = embed_weight.shape[1]
    out = torch.empty(t * h * w, hidden, device=embed_weight.device, dtype=dtype)
    _bilinear_pos_embed_kernel[(t * h * w,)](
        embed_weight,
        out,
        h,
        w,
        float(num_grid - 1) / float(h - 1) if h > 1 else 0.0,
        float(num_grid - 1) / float(w - 1) if w > 1 else 0.0,
        num_grid,
        m_size,
        hidden,
        triton.next_power_of_2(hidden),
    )
    return out


def _rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


class Qwen3VisionPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, hidden_size=1152, patch_size=14, temporal_patch_size=2):
        super().__init__()
        self.in_channels = in_channels
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        k = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels, hidden_size, kernel_size=k, stride=k)

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(x).view(-1, self.hidden_size)


class Qwen3VisionMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Qwen3VisionAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, rotary_cos, rotary_sin, seqlens, offsets, attn: ViTAttentionBackend):
        if x.dim() == 3:
            x = x.squeeze(1)
        n = x.shape[0]
        q, k, v = self.qkv(x).view(n, 3, self.num_heads, self.head_dim).unbind(1)
        cos = rotary_cos[:n].unsqueeze(1).float()
        sin = rotary_sin[:n].unsqueeze(1).float()
        qf, kf = q.float(), k.float()
        q = (qf * cos + _rotate_half(qf) * sin).to(q.dtype)
        k = (kf * cos + _rotate_half(kf) * sin).to(k.dtype)
        out = attn.run(q.contiguous(), k.contiguous(), v.contiguous())
        return self.proj(out.reshape(n, -1)).unsqueeze(1)


class Qwen3VisionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Qwen3VisionAttention(hidden_size, num_heads)
        self.mlp = Qwen3VisionMLP(hidden_size, intermediate_size)

    def forward(self, x, cos, sin, seqlens, offsets, attn):
        x = x + self.attn(self.norm1(x), cos, sin, seqlens, offsets, attn)
        return x + self.mlp(self.norm2(x))


class Qwen3VisionPatchMerger(nn.Module):
    def __init__(self, hidden_size, out_hidden_size, spatial_merge_size=2, use_postshuffle_norm=False):
        super().__init__()
        self.use_postshuffle_norm = use_postshuffle_norm
        merge = hidden_size * spatial_merge_size**2
        self.norm = nn.LayerNorm(merge if use_postshuffle_norm else hidden_size, eps=1e-6)
        self.fc1 = nn.Linear(merge, merge)
        self.fc2 = nn.Linear(merge, out_hidden_size)

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
        self.deepstack_indexes = list(getattr(c, "deepstack_visual_indexes", []) or [])
        self.rotary_head_dim = self.hidden_size // self.num_heads // 2

        self.patch_embed = Qwen3VisionPatchEmbed(
            getattr(c, "in_channels", 3),
            self.hidden_size,
            self.patch_size,
            self.temporal_patch_size,
        )
        self.pos_embed = nn.Embedding(self.num_position_embeddings, self.hidden_size)
        mid = getattr(c, "intermediate_size", 4304)
        self.blocks = nn.ModuleList(
            [
                Qwen3VisionBlock(self.hidden_size, self.num_heads, mid)
                for _ in range(getattr(c, "depth", 24))
            ]
        )
        self.merger = Qwen3VisionPatchMerger(
            self.hidden_size, self.out_hidden_size, self.spatial_merge_size
        )
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VisionPatchMerger(
                    self.hidden_size, self.out_hidden_size, self.spatial_merge_size, True
                )
                for _ in self.deepstack_indexes
            ]
        )
        self._deepstack = None
        self._attn: ViTAttentionBackend | None = None

    @property
    def deepstack_features(self):
        return self._deepstack

    def _backend(self, dtype: torch.dtype) -> ViTAttentionBackend:
        hd = self.hidden_size // self.num_heads
        if self._attn is None or self._attn.dtype != dtype or self._attn.head_dim != hd:
            self._attn = ViTAttentionBackend(
                self.num_heads, hd, self.pos_embed.weight.device, dtype
            )
        return self._attn

    def load_weights(self, state_dict: dict[str, torch.Tensor]) -> int:
        pfx = next(
            (p for p in ("model.visual.", "visual.", "") if any(k.startswith(p) for k in state_dict)),
            "",
        )
        params = dict(self.named_parameters())
        loaded = 0
        for hk, v in state_dict.items():
            if not hk.startswith(pfx):
                continue
            nk = hk[len(pfx) :].replace("linear_fc1", "fc1").replace("linear_fc2", "fc2")
            for a, b in (("attn.q.", "attn.qkv."), ("attn.k.", "attn.qkv."), ("attn.v.", "attn.qkv.")):
                nk = nk.replace(a, b)
            if nk in params:
                params[nk].data.copy_(v.to(dtype=params[nk].dtype))
                loaded += 1
        return loaded

    @staticmethod
    def _rot_pos_ids(h, w, m):
        hpos = np.broadcast_to(np.arange(h)[:, None], (h, w))
        wpos = np.broadcast_to(np.arange(w)[None, :], (h, w))
        hdiv, wdiv = h // m, w // m
        hpos = hpos.reshape(hdiv, m, wdiv, m).transpose(0, 2, 1, 3).flatten()
        wpos = wpos.reshape(hdiv, m, wdiv, m).transpose(0, 2, 1, 3).flatten()
        return torch.from_numpy(np.stack([hpos, wpos], axis=-1))

    def _rot_emb(self, grid_list):
        ids = []
        for t, h, w in grid_list:
            p = self._rot_pos_ids(h, w, self.spatial_merge_size)
            ids.append(p.repeat(t, 1) if t > 1 else p)
        pos = torch.cat(ids, 0).to(self.pos_embed.weight.device)
        inv = 1.0 / (
            10000
            ** (
                torch.arange(0, self.rotary_head_dim, 2, device=pos.device, dtype=torch.float32)
                / self.rotary_head_dim
            )
        )
        emb = torch.cat([(pos.unsqueeze(-1).float() * inv).flatten(1)] * 2, dim=-1)
        dt = self.pos_embed.weight.dtype
        return emb.cos().to(dt), emb.sin().to(dt)

    def prepare_metadata(self, grid_thw):
        grids = grid_thw.tolist() if grid_thw.dim() == 2 else grid_thw
        device = self.pos_embed.weight.device
        pos = torch.cat(
            [
                _bilinear(
                    self.pos_embed.weight,
                    t,
                    h,
                    w,
                    self.num_grid_per_side,
                    self.spatial_merge_size,
                    self.pos_embed.weight.dtype,
                )
                for t, h, w in grids
            ],
            0,
        )
        cos, sin = self._rot_emb(grids)
        g = np.asarray(grids, dtype=np.int32)
        patches = g[:, 1] * g[:, 2]
        cu = np.concatenate(
            [np.zeros(1, dtype=np.int32), np.repeat(patches, g[:, 0]).cumsum(dtype=np.int32)]
        )
        seqlens = (cu[1:] - cu[:-1]).tolist()
        return {
            "pos_embeds": pos.to(device),
            "rotary_cos": cos.to(device),
            "rotary_sin": sin.to(device),
            "cu_seqlens_cpu": torch.from_numpy(cu),
            "seqlens": seqlens,
            "offsets": cu[:-1].tolist(),
            "sizes": (g.prod(-1) // self.spatial_merge_size**2).tolist(),
        }

    def forward(self, pixel_values, grid_thw=None, metadata=None):
        dtype = self.pos_embed.weight.dtype
        device = self.pos_embed.weight.device
        md = metadata or self.prepare_metadata(grid_thw)
        hidden = self.patch_embed(pixel_values.to(device=device, dtype=dtype))
        hidden = (hidden + md["pos_embeds"]).reshape(hidden.shape[0], 1, -1)

        attn = self._backend(dtype)
        attn.plan(md["cu_seqlens_cpu"], seqlens=md["seqlens"])

        deepstack = []
        ds_set = set(self.deepstack_indexes)
        for i, blk in enumerate(self.blocks):
            hidden = blk(hidden, md["rotary_cos"], md["rotary_sin"], md["seqlens"], md["offsets"], attn)
            if i in ds_set:
                deepstack.append(self.deepstack_merger_list[self.deepstack_indexes.index(i)](hidden))

        merged = self.merger(hidden)
        self._deepstack = deepstack or None
        if grid_thw is not None:
            return tuple(merged.split(md["sizes"]))
        return (merged,)
