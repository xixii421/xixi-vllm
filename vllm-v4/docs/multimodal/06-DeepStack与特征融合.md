# 06 · DeepStack 与特征融合：视觉特征如何进入 LLM

> 对应代码：`src/models/qwen3_vl.py`（`encode_and_merge`、`_build_deepstack`）、`src/models/qwen3.py`（注入点）、`src/multimodal/vit.py`（收集）
> 目标：讲清两件事 —— ① 视觉 embedding 怎么散落到文本序列；② Qwen3-VL 的 DeepStack 机制怎么把"多层级视觉特征"注入 LLM 中间层。

---

## 1. 两种"融合"要区分

| 融合 | 位置 | 做什么 |
|------|------|--------|
| **散落（scatter）** | `encode_and_merge` | 把 ViT 输出的 `[N, out_hidden]` 视觉 embedding 覆盖到 `inputs_embeds` 的占位符位置 |
| **注入（inject）** | `Qwen3VLModel.forward` | 把 DeepStack 的 `[N, hidden]` 特征加到 LLM 前 `ndps` 层的 hidden_states 上 |

散落发生在**进入 LLM 之前**，注入发生在**LLM 内部逐层**。二者互补：散落让 LLM 第 0 层就"看到"图像，注入让 LLM 中间层"再次看到"高层视觉特征。

---

## 2. 散落（scatter）：`encode_and_merge`

### 2.1 为什么必须散落而不是拼接

**"拼接"**（concat）会把视觉 token 追加到文本后面，但这样视觉 token 只能 attend 到它之前的 token，且位置错乱。**"散落"**把视觉 embedding 放到它们在模板中的真实位置（`<|image_pad|>` 处），让 LLM 的注意力自然地在"正确的相对位置"上处理图像与上下文。

### 2.2 代码：窗口感知的散落

```210:231:src/models/qwen3_vl.py
    def encode_and_merge(self, input_ids, seqs, device):
        """Prefill: ViT encode (window + cache) → scatter embeds → MRoPE positions."""
        text = self.vl_model.embed_tokens(input_ids)
        vis = self._encode_visual(seqs, device)
        embeds = text  # scatter in-place below; embed_tokens already returns a new tensor
        off = vi = 0
        for seq in seqs:
            cs, ce = seq.num_cached_tokens, seq.num_cached_tokens + seq.num_scheduled_tokens
            for mf in seq.mm_features:
                a, b = mf.start, mf.start + mf.num_encoder_tokens
                s, e = max(a, cs), min(b, ce)
                if s < e:
                    local = off + (s - cs)
                    src = vis[vi][s - a : e - a]
                    if src.dtype != embeds.dtype or src.device != embeds.device:
                        src = src.to(dtype=embeds.dtype, device=device, non_blocking=True)
                    embeds[local : local + (e - s)].copy_(src)
                vi += 1
            off += seq.num_scheduled_tokens
        positions = mrope_positions(input_ids, seqs, self.vision_tower.merge_size, device)
        self._mrope_module().ensure_cache_size(int(getattr(positions, "max_pos", 0)) + 1)
        return embeds, self._build_deepstack(seqs, embeds.shape[0], embeds.shape[1], device, embeds.dtype), positions
```

**`s, e = max(a, cs), min(b, ce)` 是窗口感知的核心**：

- `mf.start .. mf.start + num_encoder_tokens` 是**这张图在完整 prompt 中的 token 区间**。
- `cs .. ce` 是**本次 prefill 窗口**（`num_cached_tokens .. + num_scheduled_tokens`）。
- `s, e` 取交集：**只把窗口内与图片重叠的那一段视觉 embedding 放进去**。

> **为什么要窗口感知？** chunked prefill 时，一次只处理序列的一部分。如果图片只在后面的 chunk 才被编码，前面的 chunk 根本不需要放视觉 embedding（图还没编码）。只有两者有交集时才 copy —— 这样 `encode_and_merge` 可以在任意 chunk 安全执行，而不要求"整张图必须在同一个 chunk"。

**`local = off + (s - cs)`**：`off` 是当前 sequence 在 batch 里的起始位置，`s - cs` 是图片起点在窗口内的偏移。**源** `vis[vi][s-a : e-a]`：`s-a` 是图片区间内的相对偏移。两端对齐到同一个全局 token 位置，copy 即完成。

### 2.3 `_encode_visual`：查缓存 → 收集 miss → 批量编码

```230:290:src/models/qwen3_vl.py
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
                out[i] = emb if emb.device == device else emb.to(device=device, non_blocking=True)
                if ds is not None:
                    mf.deepstack = [
                        t if t.device == device else t.to(device=device, non_blocking=True)
                        for t in ds
                    ]
                continue
            miss_i.append(i)
            miss_mf.append(mf)

        if not miss_mf:
            return out

        m = self.vision_tower.merge_size
        # Each mf.data is already pinned at the engine boundary → move to GPU
        # non_blocking per image, then cat on device (avoids CPU cat + re-pin).
        pv = torch.cat(
            [mf.data.to(device=device, non_blocking=True) for mf in miss_mf],
            0,
        )
        grids, sizes = [], []
        for mf in miss_mf:
            ...
        encoded = self.vision_tower.encode(pv, grid_thw=grid)
        flat = encoded.embeds
        dps = encoded.deepstack

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
```

**要点**：
1. **先查 `EncoderCache`**（按 `mf.uid` = 像素哈希），命中直接复用，**跳过 ViT**（第 09 讲）。
2. miss 的图**拼成一个大 batch**，一次 `vision_tower.encode(pv, grid_thw=grid)`（内部走 varlen 合批，第 04 讲；或 ViT CUDA Graph，第 10 讲）。
3. 编码结果按 `sizes` 切分回每张图，**写回 `EncoderCache`**（纯 LRU，第 09 讲）。
4. DeepStack 特征（`dps`）同样按图切分挂到 `mf.deepstack`，稍后 `_build_deepstack` 用。

---

## 3. DeepStack：多层级视觉特征注入

### 3.1 为什么需要 DeepStack

Qwen3-VL 的技术报告指出：**单一输出层的视觉特征不够** —— 浅层 ViT 特征保留细节（边缘/纹理），深层特征偏向语义。把 ViT 不同层的特征（通过各自 merger）降到 LLM 维度，注入到 LLM 前几层，让 LLM 在不同抽象层级"看"到图像，能显著提升细节感知。

### 3.2 配置驱动

```98:100:src/models/qwen3_vl.py
        self._ndps = len(getattr(config.vision_config, "deepstack_visual_indexes", []) or [])
        # Token budget (not entry count) — aligns with official EncoderCacheManager.
        self._encoder_cache = EncoderCache(max_tokens=16384)
```

`deepstack_visual_indexes` 来自 `config.vision_config`，例如 `[8, 16, 24]`（在第 8/16/24 层收集）。`_ndps` 是层数，**必须与配置严格一致**，否则 LLM 注入点和 ViT 收集点错位。

### 3.3 ViT 侧：收集

```416:424:src/multimodal/vit.py
        deepstack = []
        for i, blk in enumerate(self.blocks):
            hidden = blk(hidden, cos, sin, md["seqlens"], md["offsets"], attn)
            merger = self.deepstack_index_to_module.get(i)
            if merger is not None:
                deepstack.append(merger(hidden))
```

在每层 block 前向之后，检查该层索引是否在 `deepstack_indexes` 里，是则用**对应的 `deepstack_merger`**（`use_postshuffle_norm=True` 的 merger）把当前 hidden 降维收集。`deepstack_merger_list` 在 `__init__` 里按索引建好：

```185:196:src/multimodal/vit.py
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
```

**关键**：DeepStack 的 merger 用 `use_postshuffle_norm=True`（第 03 讲），因为 deepstack 特征要在 `_build_deepstack` 里被**切分并按 token 位置拼回**，需要先在 merger 里完成归一化到同一尺度。

### 3.4 LM 侧：注入

```160:165:src/models/qwen3.py
        for i, layer in enumerate(self.layers):
            h, residual = layer(positions, h, residual)
            if i < n_ds:
                h = h + deepstack_input_embeds[i]
```

LLM 的前 `n_ds` 层，每层 forward 之后把对应的 `deepstack_input_embeds[i]` 加到 hidden_states 上。**加法而非拼接** —— 不增加序列长度，只在特征层面叠加（DeepStack 的"无额外 context 长度"设计）。

### 3.5 中间层：`_build_deepstack`

`_build_deepstack` 把每张图缓存的 `mf.deepstack`（一个 `[ndps]` 的列表，每个是 `[该图视觉token, out_hidden]`）**拼回整条序列的 `[ndps][N, hidden]`**：

```300:324:src/models/qwen3_vl.py
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
                    src = cached[li][s - a : e - a]
                    if src.device != device or src.dtype != dtype:
                        src = src.to(device=device, dtype=dtype, non_blocking=True)
                    bufs[li][local : local + (e - s)].copy_(src)
            off += seq.num_scheduled_tokens
        return bufs
```

**要点**：
- 初始化 `bufs = [torch.zeros(n, hidden) for _ in range(ndps)]`：整条序列全零，只在图片占位符处填入 DeepStack 特征。
- 同样用 `s, e = max(a, cs), min(b, ce)` 的窗口感知逻辑，只填窗口内重叠部分。
- **文本位置的 buf 保持 0**：LLM 层 `h += 0` 对文本 token 无影响，只有视觉 token 位置被加了特征。

> **注意一个设计细节**：`_build_deepstack` 的 buffer 是全序列 `[N, hidden]`，而不是只覆盖视觉 token。这样 LLM 层的加法 `h + deepstack_input_embeds[i]` 不需要做位置索引 —— 直接用**整条序列对齐的加法**，零开销。代价是多一份 `[N, hidden]` 的显存。

---

## 4. 一个完整的数据流（DeepStack 版）

```text
                        ViT（vit.py）
   pixel_values ──▶ patch_embed ──▶ blocks[0..23]
                                      │ 在 layer 8 收集 ──▶ deepstack_merger ──▶ ds_0 [vis_tok, out_dim]
                                      │ 在 layer 16 收集 ──▶ deepstack_merger ──▶ ds_1 [vis_tok, out_dim]
                                      │ 在 layer 24 收集 ──▶ deepstack_merger ──▶ ds_2 [vis_tok, out_dim]
                                      └── merger(最后一层) ──▶ main_emb [vis_tok, out_dim]

                        encode_and_merge（qwen3_vl.py）
   main_emb ── scatter 到 inputs_embeds 占位符位置
   ds_0/1/2 ── _build_deepstack ──▶ [ndps][N, hidden] 全零+填充

                        Qwen3VLModel.forward（qwen3.py）
   inputs_embeds ──▶ layer0 ──▶ +ds[0] ──▶ layer1 ──▶ +ds[1] ──▶ layer2 ──▶ +ds[2] ──▶ ... ──▶ final_norm
```

**对应关系**：`deepstack_visual_indexes = [8, 16, 24]` → `ndps=3` → LLM 前 3 层分别注入 ds_0/1/2。**索引对应是硬编码的**，配置变了就要改代码 —— 这是第 13 讲"标准件/定制件"讨论的一个点。

---

## 5. 动手实验

1. 关掉 DeepStack（`deepstack_visual_indexes=[]`），对比 `verify_vit_vs_official.py` 的数值和显存占用。
2. 打印 `_build_deepstack` 的 buffer，验证文本位置是 0、视觉位置是特征。
3. 构造 chunked prefill 场景：图片横跨两个 chunk，验证两次 `encode_and_merge` 各自只散落窗口内部分，最终拼接正确。
4. 人为把 `deepstack_visual_indexes` 的收集层改成错误的（比如收集层和注入层数量不一致），观察报错/错乱。

---

## 6. 面试 / 教学提问

- 为什么视觉 embedding 要"散落"而不是"拼接"在文本后面？
- `s, e = max(a, cs), min(b, ce)` 在 chunked prefill 下保证什么？
- DeepStack 相比"只用最后一层特征"的优势是什么？
- 为什么 DeepStack 用加法注入而不是 concat？（不增加序列长度）
- `_build_deepstack` 为什么要先分配全零 buffer 再填视觉位置？

---

**下一讲**：[07-权重加载与算子融合](07-权重加载与算子融合.md) —— HF 权重如何映射到本地的融合算子。
