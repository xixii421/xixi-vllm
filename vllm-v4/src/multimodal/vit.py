import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.vision import ViTAttentionBackend


def _bilinear(embed_weight, t, h, w, num_grid, m_size, dtype):
    assert h % m_size == 0 and w % m_size == 0
    device = embed_weight.device
    side = num_grid

    h_grid = torch.linspace(0, side - 1, h, device=device)
    w_grid = torch.linspace(0, side - 1, w, device=device)
    h_floor = h_grid.int()
    w_floor = w_grid.int()
    h_ceil = (h_floor + 1).clamp(max=side - 1)
    w_ceil = (w_floor + 1).clamp(max=side - 1)
    h_frac = h_grid - h_floor
    w_frac = w_grid - w_floor
    h_floor_offset = h_floor * side
    h_ceil_offset = h_ceil * side

    corner_indices = [
        (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
        (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
        (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
        (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
    ]
    corner_weights = [
        ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
        ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
        (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
        (h_frac[:, None] * w_frac[None, :]).flatten(),
    ]

    h_idx = torch.arange(h, device=device).view(h // m_size, m_size)
    w_idx = torch.arange(w, device=device).view(w // m_size, m_size)
    reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).transpose(1, 2).flatten()
    if t > 1:
        reorder = reorder.repeat(t)

    indices = torch.stack([c[reorder] for c in corner_indices])  # [4, t*h*w]
    weights = torch.stack([c[reorder] for c in corner_weights])  # [4, t*h*w]
    # Match HF: bf16 embed * float weights → float accumulate → cast.
    out = (embed_weight[indices].float() * weights.float().unsqueeze(-1)).sum(0)
    return out.to(dtype=dtype)


def _rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def _normalize_grids(grid_thw):
    if isinstance(grid_thw, torch.Tensor):
        if grid_thw.dim() == 1:
            grid_thw = grid_thw.view(1, 3)
        return grid_thw.tolist()
    grids = list(grid_thw)
    if grids and isinstance(grids[0], (int, float)):
        grids = [grids]
    return grids


class Qwen3VisionPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, hidden_size=1152, patch_size=16, temporal_patch_size=2):
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
    def __init__(self, hidden_size, intermediate_size, hidden_act: str = "gelu_pytorch_tanh"):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        # Qwen3-VL vision_config.hidden_act == gelu_pytorch_tanh
        self._approx = "tanh" if "tanh" in (hidden_act or "") else "none"

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate=self._approx))


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
    def __init__(self, hidden_size, num_heads, intermediate_size, hidden_act: str = "gelu_pytorch_tanh"):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Qwen3VisionAttention(hidden_size, num_heads)
        self.mlp = Qwen3VisionMLP(hidden_size, intermediate_size, hidden_act)

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
        # Official merger uses nn.GELU() (exact), not tanh approx.
        self.act = nn.GELU()

    def forward(self, x):
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.fc1.in_features))
        else:
            x = self.norm(x).view(-1, self.fc1.in_features)
        return self.fc2(self.act(self.fc1(x)))


class Qwen3VisionTransformer(nn.Module):
    def __init__(self, vision_config):
        super().__init__()
        c = vision_config
        self.hidden_size = c.hidden_size
        self.num_heads = c.num_heads
        # Qwen3-VL default patch_size is 16 (not Qwen2-VL's 14).
        self.patch_size = getattr(c, "patch_size", 16)
        self.temporal_patch_size = getattr(c, "temporal_patch_size", 2)
        self.spatial_merge_size = getattr(c, "spatial_merge_size", 2)
        self.num_position_embeddings = getattr(c, "num_position_embeddings", 2304)
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)
        self.out_hidden_size = getattr(c, "out_hidden_size", c.hidden_size)
        self.deepstack_indexes = list(getattr(c, "deepstack_visual_indexes", []) or [])
        self.rotary_head_dim = self.hidden_size // self.num_heads // 2
        hidden_act = getattr(c, "hidden_act", "gelu_pytorch_tanh")

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
                Qwen3VisionBlock(self.hidden_size, self.num_heads, mid, hidden_act)
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
        self.deepstack_index_to_module = {
            idx: module
            for idx, module in zip(self.deepstack_indexes, self.deepstack_merger_list)
        }
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
        """Load HF / fused / split-QKV vision weights into this module."""
        pfx = next(
            (p for p in ("model.visual.", "visual.", "") if any(k.startswith(p) for k in state_dict)),
            "",
        )
        params = dict(self.named_parameters())
        loaded = 0
        hs = self.hidden_size

        def _copy_slice(target_key: str, src: torch.Tensor, start: int, end: int) -> bool:
            if target_key not in params:
                return False
            dst = params[target_key]
            params[target_key].data[start:end].copy_(src.to(dtype=dst.dtype))
            return True

        for hk, v in state_dict.items():
            if not hk.startswith(pfx):
                continue
            nk = hk[len(pfx) :].replace("linear_fc1", "fc1").replace("linear_fc2", "fc2")

            # Split Q/K/V → fused qkv (shard copy). Check before generic path.
            if nk.endswith("attn.q.weight"):
                t = nk.replace("attn.q.weight", "attn.qkv.weight")
                loaded += int(_copy_slice(t, v, 0, hs))
                continue
            if nk.endswith("attn.k.weight"):
                t = nk.replace("attn.k.weight", "attn.qkv.weight")
                loaded += int(_copy_slice(t, v, hs, 2 * hs))
                continue
            if nk.endswith("attn.v.weight"):
                t = nk.replace("attn.v.weight", "attn.qkv.weight")
                loaded += int(_copy_slice(t, v, 2 * hs, 3 * hs))
                continue
            if nk.endswith("attn.q.bias"):
                t = nk.replace("attn.q.bias", "attn.qkv.bias")
                loaded += int(_copy_slice(t, v, 0, hs))
                continue
            if nk.endswith("attn.k.bias"):
                t = nk.replace("attn.k.bias", "attn.qkv.bias")
                loaded += int(_copy_slice(t, v, hs, 2 * hs))
                continue
            if nk.endswith("attn.v.bias"):
                t = nk.replace("attn.v.bias", "attn.qkv.bias")
                loaded += int(_copy_slice(t, v, 2 * hs, 3 * hs))
                continue

            if nk in params:
                params[nk].data.copy_(v.to(dtype=params[nk].dtype))
                loaded += 1
        return loaded

    @staticmethod
    @functools.lru_cache(maxsize=1024)
    def _rot_pos_ids(h: int, w: int, m: int):
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
        pos = torch.cat(ids, 0).to(self.pos_embed.weight.device, non_blocking=True)
        inv = 1.0 / (
            10000
            ** (
                torch.arange(0, self.rotary_head_dim, 2, device=pos.device, dtype=torch.float32)
                / self.rotary_head_dim
            )
        )
        emb = torch.cat([(pos.unsqueeze(-1).float() * inv).flatten(1)] * 2, dim=-1)
        # Keep cos/sin in float32 — matches HF VisionAttention (casts to float before apply).
        return emb.cos(), emb.sin()

    def prepare_metadata(
        self,
        grid_thw,
        *,
        max_num_seqs: int | None = None,
        max_seqlen_override: int | None = None,
        max_patches: int | None = None,
    ):
        """Build encoder metadata.

        ``max_num_seqs`` / ``max_seqlen_override`` / ``max_patches`` mirror
        official vLLM ``prepare_encoder_metadata`` padding for CUDA Graph:
        cu_seqlens is padded with zero-length seqs; max_seqlen is baked at
        capture; patch buffers may be larger than ``cu_seqlens[-1]`` (FA
        ignores tokens beyond the last offset).
        """
        grids = _normalize_grids(grid_thw)
        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype
        pos = torch.cat(
            [
                _bilinear(
                    self.pos_embed.weight,
                    t,
                    h,
                    w,
                    self.num_grid_per_side,
                    self.spatial_merge_size,
                    dtype,
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
        if max_num_seqs is not None and (len(cu) - 1) < max_num_seqs:
            cu = np.concatenate(
                [cu, np.full(max_num_seqs - (len(cu) - 1), cu[-1], dtype=np.int32)]
            )
        seqlens = (cu[1:] - cu[:-1]).tolist()
        max_seqlen = (
            int(max_seqlen_override)
            if max_seqlen_override is not None
            else (max(seqlens) if seqlens else 0)
        )
        # Optional pad of pos/rope to max_patches for static CG buffers.
        if max_patches is not None and pos.shape[0] < max_patches:
            pad_n = max_patches - pos.shape[0]
            pos = torch.cat([pos, pos.new_zeros(pad_n, pos.shape[1])], 0)
            cos = torch.cat([cos, cos.new_zeros(pad_n, cos.shape[1])], 0)
            sin = torch.cat([sin, sin.new_zeros(pad_n, sin.shape[1])], 0)
        return {
            "pos_embeds": pos.to(device, non_blocking=True),
            "rotary_cos": cos.to(device, non_blocking=True),
            "rotary_sin": sin.to(device, non_blocking=True),
            "cu_seqlens_cpu": torch.from_numpy(cu.copy()),
            "cu_seqlens": torch.from_numpy(cu.copy()).to(
                device=device, dtype=torch.int32, non_blocking=True
            ),
            "seqlens": seqlens,
            "offsets": cu[:-1].tolist(),
            "sizes": (g.prod(-1) // self.spatial_merge_size**2).tolist(),
            "max_seqlen": max_seqlen,
            "num_patches": int((g[:, 0] * g[:, 1] * g[:, 2]).sum()),
        }

    def forward(
        self,
        pixel_values,
        grid_thw=None,
        metadata=None,
        *,
        concat_deepstack: bool = False,
        plan_attn: bool = True,
        split_images: bool = True,
    ):
        """ViT forward.

        Args:
            concat_deepstack: If True, return ``[N, out_dim * (1 + n_deepstack)]``
                like official vLLM encoder CUDA Graph output.
            plan_attn: If False, attention must already be ``bind``-ed to the
                static ``metadata['cu_seqlens']`` GPU buffer (CG path).
            split_images: If True and sizes known, return per-image tuple of
                merged embeds (eager API). Ignored when ``concat_deepstack``.
        """
        dtype = self.pos_embed.weight.dtype
        device = self.pos_embed.weight.device
        if metadata is None:
            if grid_thw is None:
                raise ValueError("forward requires grid_thw or metadata")
            md = self.prepare_metadata(grid_thw)
        else:
            md = metadata

        pv = pixel_values
        if pv.device != device or pv.dtype != dtype:
            pv = pv.to(device=device, dtype=dtype, non_blocking=True)
        # CG buffers may be longer than the active prefix; FA uses cu_seqlens.
        hidden = self.patch_embed(pv)
        n = hidden.shape[0]
        pos = md["pos_embeds"]
        cos, sin = md["rotary_cos"], md["rotary_sin"]
        # Static CG buffers are sized to max_patches; active prefix may be
        # shorter in metadata tensors before pad — align to hidden length.
        if pos.shape[0] < n:
            raise RuntimeError(
                f"pos_embeds length {pos.shape[0]} < patches {n}; "
                "pass max_patches= when building CG metadata"
            )
        if pos.shape[0] > n:
            pos, cos, sin = pos[:n], cos[:n], sin[:n]
        hidden = (hidden + pos).reshape(n, 1, -1)

        attn = self._backend(dtype)
        if plan_attn:
            attn.plan(md["cu_seqlens_cpu"], seqlens=md["seqlens"])
        # else: caller already attn.bind()'d a static cu_seqlens buffer (CG path).

        deepstack = []
        for i, blk in enumerate(self.blocks):
            hidden = blk(hidden, cos, sin, md["seqlens"], md["offsets"], attn)
            merger = self.deepstack_index_to_module.get(i)
            if merger is not None:
                deepstack.append(merger(hidden))

        merged = self.merger(hidden)
        self._deepstack = deepstack or None

        if concat_deepstack:
            if deepstack:
                return torch.cat([merged] + deepstack, dim=1)
            return merged

        if split_images and (grid_thw is not None or "sizes" in md):
            return tuple(merged.split(md["sizes"]))
        return (merged,)
