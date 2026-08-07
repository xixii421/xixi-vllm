# 03 · 视觉编码器 ViT：patch 如何变成 token

> 对应代码：`src/multimodal/vit.py`
> 目标：从零读懂 `Qwen3VisionTransformer` —— patch embedding、可学习位置编码（含双线性插值）、transformer 主干、PatchMerger。

---

## 1. ViT 在做什么

ViT（Vision Transformer）把一张图变成一串向量。Qwen3-VL 的 ViT 输入是 `pixel_values`（`[T*H*W, C*2*16*16]` 的 patch 展平），输出是每个 patch 的 `hidden_size` 维向量：

```text
输入:  pixel_values   [num_patches, in_ch*2*16*16]   (展平的 patch)
输出:  hidden         [num_patches, hidden_size]     (每个 patch 一个向量)
```

```
┌─────────────────────────────────────────────────────────┐
│ Qwen3VisionTransformer                                   │
│  patch_embed  (Conv3d)    展平像素 → patch token        │
│  + pos_embed  (双线性插值) 加上 2D 位置编码               │
│  blocks × depth           Transformer 主干               │
│  merger       (PatchMerger) 2×2 patch → 1 个 token       │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 模块逐层拆解

### 2.1 `Qwen3VisionPatchEmbed`：像素 → patch token

```69:81:src/multimodal/vit.py
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
```

**关键点：这是一个 `Conv3d`，kernel = `(temporal_patch_size, patch_size, patch_size)`，stride 相同。**

- Qwen3-VL 的默认 `patch_size = 16`（注意：**不是 Qwen2-VL 的 14**）。
- `temporal_patch_size = 2`：把连续 2 帧在时间维合并。图片 t=1，所以输入 `pixel_values` 被 view 成 `[1, C, 2, 16, 16]` 的一个 patch 单元。
- 输出 `[num_patches, hidden_size]`：每个 patch 变成一个向量。

> **和纯文本对比**：`embed_tokens` 是查表（embedding lookup），ViT 的 `patch_embed` 是卷积。一张图先被切成 `t×h×w` 个 patch，每个 patch 再过卷积变成向量 —— 这就是"图像 token 化"。

### 2.2 位置编码 `pos_embed`：双线性插值

ViT 的 `pos_embed` 是一个 `nn.Embedding(num_position_embeddings, hidden_size)`。Qwen3-VL 训练时位置网格是固定的 `num_grid_per_side × num_grid_per_side`（默认 `2304 → 48×48`）。**推理时图像分辨率不固定，patch 网格也不同**（比如 14×14），所以不能直接查表，而要**用双线性插值把训练网格缩放到当前网格**：

```311:325:src/multimodal/vit.py
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
```

`_bilinear` 的核心思想：

```11:50:src/multimodal/vit.py
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
```

- 把训练网格的 `[0, num_grid-1]` 均匀映射到当前 `[0, h-1]` / `[0, w-1]`。
- 每个当前网格点取四个最近邻训练格点，按距离加权平均（双线性）。

```46:50:src/multimodal/vit.py
    # Match HF: bf16 embed * float weights → float accumulate → cast.
    out = (embed_weight[indices].float() * weights.float().unsqueeze(-1)).sum(0)
    return out.to(dtype=dtype)
```

**注意精度细节**：`embed_weight` 是 bf16，权重是 float32，**先 cast float 再乘加，最后 cast 回原 dtype** —— 这与 HF 官方实现严格对齐（`verify_vit_vs_official.py` 验证过 bit-exact）。

### 2.3 `Qwen3VisionBlock`：Transformer 主干

```119:129:src/multimodal/vit.py
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
```

这就是标准 ViT block：`x = x + Attn(LN(x))`，再 `x = x + MLP(LN(x))`。**ViT 的 attention 是双向的（不 causal）**，这一点和 LLM 完全不同，具体在第 04 讲展开。

#### 2.3.1 `Qwen3VisionMLP`：GELU 的近似差异

```84:93:src/multimodal/vit.py
class Qwen3VisionMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act: str = "gelu_pytorch_tanh"):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        # Qwen3-VL vision_config.hidden_act == gelu_pytorch_tanh
        self._approx = "tanh" if "tanh" in (hidden_act or "") else "none"

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate=self._approx))
```

**一个容易踩的坑**：Qwen3-VL 视觉 MLP 用的是 `gelu_pytorch_tanh`（tanh 近似 GELU），而 PatchMerger 用的是精确 `nn.GELU()`（见下文）—— **同一个模型里两种 GELU 同时存在**。如果统一用精确 GELU 会 bit-exact 失败。

#### 2.3.2 `Qwen3VisionAttention`：融合 QKV + 双向 attention

```96:116:src/multimodal/vit.py
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
```

要点：
- **QKV 融合**为一个 `Linear(hidden, 3*hidden)`（`qkv`），一次 GEMM 出 Q/K/V，而不是三个独立 Linear —— 这是权重融合（第 07 讲讲权重加载时这里对应"split Q/K/V → fused qkv"）。
- **Q/K 应用旋转位置编码**（ViT 内部自己的 2D RoPE，不是 LLM 的 MRoPE），用 `rotary_cos/sin`（由 `_rot_emb` 预计算）。
- **`attn.run(...)` 调外部后端**（`ViTAttentionBackend`，第 04 讲），`seqlens/offsets` 只传给后端，这里不直接用。

#### 2.3.3 ViT 内部的旋转位置编码（`_rot_emb`）

```265:290:src/multimodal/vit.py
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
```

- **`_rot_pos_ids`**：对 `h×w` 网格生成每个 patch 的 2D 坐标 `(h_pos, w_pos)`。注意它做了 `merge_size` 的 reshape/transpose —— **坐标的展平顺序与 PatchMerger 的合并顺序一致**（这保证了 merge 后 token 的顺序与位置一致，是正确性细节）。
- **`_rot_emb`**：把 2D 坐标做成 RoPE 的 cos/sin。`inv` 是标准 `1/10000^(i/d)` 频率表，`rotary_head_dim = hidden/num_heads/2` 表示只有一半的 head_dim 参与旋转。
- **`lru_cache`**：`_rot_pos_ids` 对相同的 `(h,w,m)` 缓存，不同分辨率不会重复计算。

### 2.4 `Qwen3VisionPatchMerger`：2×2 降维

```132:148:src/multimodal/vit.py
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
```

**PatchMerger 是 Qwen-VL 系列的标志性结构**：把空间相邻的 `spatial_merge_size×spatial_merge_size`（默认 2×2）个 patch token 合并成一个 token。

```text
[hidden]×4 个相邻 patch
   │  reshape → [1, hidden*4]
   │  fc1 (Linear: hidden*4 → hidden*4)
   │  GELU
   │  fc2 (Linear: hidden*4 → out_hidden_size)
   ▼
[out_hidden] 一个 token
```

- `merge = hidden_size * 4`：4 个 patch 拼成一维再线性变换。
- **`use_postshuffle_norm`**：控制 norm 顺序。默认 merger 是 `norm(x)` 先对整个序列做 LayerNorm 再 view；deepstack merger 用 `use_postshuffle_norm=True`，先 view 成 `[N/4, hidden*4]` 再 norm（因为 deepstack merger 输入已经是 flatten 过的特征）。
- **`fc1` 的输入特征数其实是 `merge`**（`self.fc1 = nn.Linear(merge, merge)`），所以 view 目标 `self.fc1.in_features` 正好等于 `merge`。

> **为什么输出是 `out_hidden_size` 而不是 `hidden_size`？** Qwen3-VL 的视觉塔输出维度 `out_hidden_size` 可以比 `hidden_size` 大（deepstack 场景），要和 LLM 的 `hidden_size` 对齐，所以 merger 的最后一层 `fc2` 负责维度对齐。这样 LLM 才能直接把视觉 embedding 和文本 embedding 相加/拼接。

---

## 3. 前向流程 `forward()`

```363:433:src/multimodal/vit.py
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
        ...
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
```

**流程拆解：**

1. **取元数据**：`metadata`（预计算的 pos_embeds/cos/sin/cu_seqlens）由 `prepare_metadata()` 一次性算好，或在 forward 内部现算。**这是为 CUDA Graph 埋的关键伏笔**：CG 路径把 `metadata` 预先构造好并绑定静态 buffer，forward 里不重新算。
2. **对齐 buffer**：`pixel_values` 可能比 `metadata` 短（CG 静态 buffer 比实际长），所以 `pos/cos/sin` 按 `n = hidden.shape[0]` 截断。
3. **加位置编码**：`hidden = patch_embed(pv) + pos_embed`，然后 `reshape(n, 1, -1)`（保留单头维度给 attention）。
4. **attention 规划**：`plan_attn=True` 时把 `cu_seqlens` 拷到 GPU（`attn.plan`）；CG 路径 `plan_attn=False` 直接走 `bind`（第 04/10 讲）。
5. **逐 block 前向 + 收集 DeepStack**：在第 `deepstack_indexes` 指定的层（如 `[8, 16, 24]`）处，用对应 `deepstack_merger` 把隐藏状态降维收集进 `deepstack`。
6. **主路 merger**：最后一层输出过主 `merger`，得到每个视觉 token 的最终 embedding。
7. **输出形态**：`concat_deepstack=True` 时输出 `[N, out_dim*(1+ndps)]`（CG 路径专用）；否则按图像边界 `split` 成 per-image tuple（eager 路径专用）。

---

## 4. 为什么 ViT 的 forward 要留这么多 `*` 参数开关

`concat_deepstack` / `plan_attn` / `split_images` 三个开关的存在，是因为**同一个 forward 要服务三种调用方**：

| 调用方 | concat_deepstack | plan_attn | split_images |
|--------|:---:|:---:|:---:|
| eager 单图/多图（`_encode_eager`） | False | True | True |
| CUDA Graph 捕获（`VisionEncoderCudaGraph.capture`） | True | False | False |
| CUDA Graph 回放（`try_encode`） | True | False | False |

**这是"一份代码、三种形态"的设计**：eager 路径和 CG 路径共享同一个 forward，只是通过开关切换。这也是第 10 讲能实现 ViT CUDA Graph 的前提 —— 如果 eager 和 CG 是两套实现，维护成本翻倍且容易产生行为不一致。

---

## 5. 动手实验

1. 单张图走 eager：打印 `prepare_metadata()` 的 `pos_embeds`/`rotary_cos` shape，对比 `pixel_values` shape，确认三者都等于 `num_patches`。
2. 把 `_bilinear` 里的 `embed_weight.float()` 去掉（直接 `embed_weight[indices] * weights`），对比 `verify_vit_vs_official.py` 是否还能 bit-exact。
3. 修改 `_rot_emb` 的 `rotary_head_dim` 计算（比如改成全 `hidden_size`），观察输出 shape 与 correctness 变化。
4. 用 `use_postshuffle_norm=True` 的 merger 处理一个非 merge-size 整数倍的序列，观察 `view` 的报错，理解"为什么 patch 数必须是 `merge_size²` 的倍数"。

---

## 6. 面试 / 教学提问

- `patch_embed` 为什么用 `Conv3d` 而不是 `Conv2d`？`temporal_patch_size` 在图片场景的值是多少？
- ViT 的位置编码为什么需要双线性插值？训练网格和推理网格不一致时直接查表会怎样？
- `Qwen3VisionAttention` 的 QKV 为什么融合成一个 `Linear`？
- PatchMerger 的 `spatial_merge_size=2` 把 token 数降到多少？和 MRoPE 的 grid 有什么关系？
- 为什么 ViT 的 forward 要为 eager/CG 两种路径保留三个开关？

---

**下一讲**：[04-ViT注意力与合批](04-ViT注意力与合批.md) —— 多张图怎么用 varlen FlashAttention 一次性算完。
