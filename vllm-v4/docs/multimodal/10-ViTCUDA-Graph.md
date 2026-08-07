# 10 · ViT CUDA Graph：变长输入如何用预算式 CUDA Graph 加速

> 对应代码：`src/multimodal/vision.py`（`VisionEncoderCudaGraph`、`_BudgetGraph`）、`src/multimodal/vit.py`（`prepare_metadata`）
> 目标：讲清 CUDA Graph 为什么对 ViT 重要、预算式（budget）捕获怎么解决"变长输入"问题、`bind` vs `plan` 在捕获中的角色。

---

## 1. 为什么 ViT 需要 CUDA Graph

ViT 的 forward 由大量**小 kernel** 组成：patch embed（卷积）、位置编码（gather）、RoPE、LayerNorm、QKV GEMM、attention、MLP…… 每个 kernel 都有 CPU 侧 launch 开销（~2-10μs）。

单张中等图约 2000 个视觉 token，ViT 有 24+ 层，每层 2 个 block（attn + mlp），粗算几十到上百次 kernel launch。**如果每张图都走 eager，CPU 发射 GPU 的延迟会成为吞吐瓶颈** —— 这正是"为什么 ViT 慢"的一个核心原因。

**CUDA Graph 解决方式**：一次捕获（capture）整张计算图，之后每次只需 `graph.replay()`（一次 launch 重放整图），launch 开销从 O(几十次) 降到 O(1)。

```text
eager:  CPU ─launch─▶ kernel1 ─▶ kernel2 ─▶ ... ─▶ kernelN   （N 次 CPU 往返）
CG:     CPU ─replay──▶ [GPU 上整张图串行执行]                    （1 次 CPU 往返）
```

---

## 2. 核心难点：ViT 输入是变长的

纯文本 decode 的输入是**固定 batch**（`max_batch_size`），所以 v3 的 decode CG 直接按最大 batch 捕获。但 **ViT 的输入 token 数（patch 数）随图像分辨率变化** —— 一张 512×512 图和一张 1024×1024 图，patch 数差 4 倍。

CUDA Graph 要求捕获时**所有内存地址和 kernel 参数固定**。变长输入意味着：
- `cu_seqlens` 长度不同（图数量不同）。
- `max_seqlen` 不同（最大图的 patch 数不同）。
- `pos_embeds` / `rotary_cos/sin` / `pixel_values` 长度不同。

**预算式（budget）设计**就是为变长而生的：捕获**多档固定大小的图**（budget = 视觉 token 数上限），运行时选一档能装下当前输入的去重放。

---

## 3. 预算式捕获：`VisionEncoderCudaGraph.capture`

### 3.1 捕获骨架

```168:263:src/multimodal/vision.py
    def capture(self) -> None:
        device = self.vit.pos_embed.weight.device
        dtype = self.vit.pos_embed.weight.dtype
        m = self.vit.spatial_merge_size
        pe = self.vit.patch_embed
        flat_ps = pe.in_channels * pe.temporal_patch_size * pe.patch_size * pe.patch_size
        out_dim = self.vit.out_hidden_size
        n_ds = len(self.vit.deepstack_indexes)
        out_dim_cat = out_dim * (1 + n_ds)

        self.graphs.clear()
        self._graph_pool = torch.cuda.graph_pool_handle()
        # Warmup backend once outside capture.
        _ = self.vit._backend(dtype)

        for budget in sorted(self.token_budgets, reverse=True):
            if budget < self.max_batch_size:
                continue
            per_item = (budget + self.max_batch_size - 1) // self.max_batch_size
            grid_config = [
                [1, m, per_item * m] for _ in range(self.max_batch_size)
            ]
            max_patches = sum(t * h * w for t, h, w in grid_config)
            max_seqlen = budget * (m * m)
            ...
```

**`token_budgets`** 默认 `[64, 128, 256, 512, 1024, 2048]` —— 这是 **merge 后视觉 token 数**的档位。

**关键：捕获时怎么构造"dummy 输入"？** 对每个 budget：
1. 把 budget 平分给 `max_batch_size` 张"假图"（`per_item = ceil(budget / max_batch)`）。
2. 每张假图的 `grid_config = [1, m, per_item * m]`（即 `t=1, h=m, w=per_item*m`），patch 数 `m * per_item*m = per_item * m²`，merge 后正好 `per_item` 个 token。这样 `max_patches` 有确定值。
3. `max_seqlen = budget * m²`：捕获时固定传给 `flash_attn_varlen_func`。

**`max_batch_size` 的作用**：一个预算档能处理"最多 `max_batch_size` 张图，总 token ≤ budget"。多张图共享一个大 budget，而不是每张图单独一个 budget。

### 3.2 捕获中的 forward 调用

```223:242:src/multimodal/vision.py
            with torch.inference_mode():
                out = self.vit(
                    buffers["pixel_values"],
                    metadata=meta,
                    concat_deepstack=True,
                    plan_attn=False,
                    split_images=False,
                )
                tmp = torch.empty_like(out)

            g = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(g, pool=self._graph_pool):
                out = self.vit(
                    buffers["pixel_values"],
                    metadata=meta,
                    concat_deepstack=True,
                    plan_attn=False,
                    split_images=False,
                )
                tmp.copy_(out)
```

**两次 forward**：第一次是"预热"（warmup），让所有 CUDA kernel 缓存/初始化完成；第二次才真正 `torch.cuda.graph(g)` 捕获。**为什么必须预热？** CUDA Graph 捕获期间不能有隐式分配；预热让所有内存分配、cuBLAS workspace、FlashAttention workspace 都在捕获前稳定。

**三个开关**（第 03 讲提过）：
- `concat_deepstack=True`：输出拼成 `[N, out_dim*(1+ndps)]` 单张量，方便 `copy_` 到固定输出 buffer。
- `plan_attn=False`：**不在 forward 里调用 `attn.plan`（有 H2D 拷贝，捕获期禁止）**。捕获前已用 `bind` 绑定静态 cu_seqlens。
- `split_images=False`：不切分输出（输出 buffer 是整块的）。

### 3.3 `bind` 的不可替代性

```220:222:src/multimodal/vision.py
            attn = self.vit._backend(dtype)
            attn.bind(buffers["cu_seqlens"], max_seqlen)
```

捕获前把 GPU 上的静态 `cu_seqlens` buffer 绑给 backend。**捕获时 kernel 用的就是这个固定地址**。如果捕获期间走 `plan`（`to(device)`），就会在捕获流里插入一次 H2D 拷贝 —— CUDA Graph 的捕获对 `to()` 这类 API 是**不允许**的（它可能触发隐式分配）。

> **这呼应了第 04 讲的 `plan` vs `bind` 设计**：`bind` 就是专门为 CG 捕获而存在的静态绑定入口。

---

## 4. 运行时回放：`try_encode`

```279:328:src/multimodal/vision.py
    def try_encode(
        self, pixel_values: torch.Tensor, grid_thw
    ) -> VisionEncodeOutput | None:
        if not self.enabled:
            return None
        grids = _normalize_grids(grid_thw)
        m = self.vit.spatial_merge_size
        sizes = [int(t * h * w) // (m * m) for t, h, w in grids]
        total = sum(sizes)
        budget = self._find_budget(total, len(grids))
        if budget is None:
            self.misses += 1
            return None

        bg = self.graphs[budget]
        # Build real metadata (padded seq count to max_batch).
        md = self.vit.prepare_metadata(
            grids,
            max_num_seqs=bg.max_batch_size,
            max_seqlen_override=bg.max_seqlen,
            max_patches=bg.max_patches,
        )
        bufs = bg.input_buffers
        self._copy_pad(bufs["pixel_values"], pixel_values)
        self._copy_pad(bufs["pos_embeds"], md["pos_embeds"])
        self._copy_pad(bufs["rotary_cos"], md["rotary_cos"])
        self._copy_pad(bufs["rotary_sin"], md["rotary_sin"])
        # cu_seqlens: copy then ensure length matches static buffer.
        cu_dst = bufs["cu_seqlens"]
        cu_src = md["cu_seqlens"]
        cu_dst.zero_()
        cu_dst[: cu_src.numel()].copy_(cu_src)
        if cu_src.numel() < cu_dst.numel():
            # Pad remaining slots with last offset (zero-length seqs).
            cu_dst[cu_src.numel() :].fill_(int(cu_src[-1].item()))

        bg.graph.replay()
        self.hits += 1

        out = bg.output_buffer[:total]
        ...
```

### 4.1 三步回放

1. **`_find_budget(total, num_items)`**：选一个"能装下当前输入"的最小 budget。
2. **`prepare_metadata(..., max_num_seqs, max_seqlen_override, max_patches)`**：用真实输入重算元数据，但**填充到捕获时的固定尺寸**（第 04 讲的三个 padding 参数）。
3. **`_copy_pad` 填静态 buffer → `graph.replay()`**：把真实 pixel_values / pos_embeds / cos / sin / cu_seqlens 拷贝进捕获时的静态 buffer（`copy_` 在捕获外的流里做，合法），然后一次 replay。

### 4.2 `_find_budget`：装得下的最小档

```265:271:src/multimodal/vision.py
    def _find_budget(self, total_out_tokens: int, num_items: int) -> int | None:
        if num_items > self.max_batch_size:
            return None
        for b in self.token_budgets:
            if b >= total_out_tokens and b in self.graphs:
                return b
        return None
```

- 图数量超过 `max_batch_size` → 返回 None（放不下，走 eager）。
- 否则选 `>= total` 的**最小** budget（`token_budgets` 已排序）。

**`_copy_pad` 处理"缓冲比真实输入长"**：

```273:277:src/multimodal/vision.py
    @staticmethod
    def _copy_pad(dst: torch.Tensor, src: torch.Tensor) -> None:
        dst.zero_()
        n = min(dst.shape[0], src.shape[0])
        dst[:n].copy_(src[:n])
```

静态 buffer 比真实输入长，多余的清 0。**cu_seqlens 的 padding 尤其关键**：

```306:313:src/multimodal/vision.py
        cu_dst = bufs["cu_seqlens"]
        cu_src = md["cu_seqlens"]
        cu_dst.zero_()
        cu_dst[: cu_src.numel()].copy_(cu_src)
        if cu_src.numel() < cu_dst.numel():
            # Pad remaining slots with last offset (zero-length seqs).
            cu_dst[cu_src.numel() :].fill_(int(cu_src[-1].item()))
```

多余槽位填 `cu_seqlens[-1]`（最后一个真实 offset）—— 让"多余的假图"表现为 **zero-length seq**（起点 = 终点 = 最后一个真实 offset），varlen FA 安全跳过。**不变式：`cu_seqlens[-1] ≤ max_patches`，保证 FA 不越界。**

### 4.3 回放后的切分

```318:328:src/multimodal/vision.py
        out = bg.output_buffer[:total]
        out_dim = self.vit.out_hidden_size
        n_ds = len(self.vit.deepstack_indexes)
        if n_ds:
            chunks = out.split(out_dim, dim=1)
            embeds = chunks[0].contiguous()
            deepstack = [c.contiguous() for c in chunks[1:]]
        else:
            embeds = out.contiguous()
            deepstack = None
        return VisionEncodeOutput(embeds=embeds, deepstack=deepstack, sizes=sizes)
```

输出 buffer 是 `[max_patches, out_dim*(1+ndps)]`，截取前 `total` 个 token（真实 token 数），再按列切分主 embedding 和 DeepStack。

---

## 5. 回退与贪心打包：`Qwen3VisionTower.encode`

```378:419:src/multimodal/vision.py
    @torch.inference_mode()
    def encode(
        self,
        pixel_values: torch.Tensor,
        grid_thw,
        *,
        use_cudagraph: bool | None = None,
    ) -> VisionEncodeOutput:
        """Batch-encode images; prefer CUDA Graph when warmed and shapes fit."""
        assert self.vit is not None
        grids = _normalize_grids(grid_thw)
        m = self.merge_size
        sizes = [int(t * h * w) // (m * m) for t, h, w in grids]
        use_cg = self._use_cudagraph if use_cudagraph is None else use_cudagraph
        patch_counts = [int(t * h * w) for t, h, w in grids]

        if use_cg and self.cudagraph is not None and self.cudagraph.enabled:
            hit = self.cudagraph.try_encode(pixel_values, grids)
            if hit is not None:
                return hit
            # Greedy pack: run CG per item (or small groups) when batch is too wide.
            if len(grids) > 1:
                embeds_l, ds_l = [], []
                off = 0
                for g, pc, sz in zip(grids, patch_counts, sizes):
                    pv_i = pixel_values[off : off + pc]
                    off += pc
                    sub = self.cudagraph.try_encode(pv_i, [g])
                    if sub is None:
                        sub = self._encode_eager(pv_i, [g], [sz])
                    embeds_l.append(sub.embeds)
                    ...
                embeds = torch.cat(embeds_l, dim=0)
                ...
                return VisionEncodeOutput(embeds=embeds, deepstack=deepstack, sizes=sizes)

        return self._encode_eager(pixel_values, grids, sizes)
```

**两级回退**：
1. **整体 CG**：所有图拼一起能装进一个 budget → `try_encode` 一次搞定。
2. **贪心打包**：batch 太宽（图数超 `max_batch_size`）→ **逐张（或小组）走 CG**，每张选自己的 budget；某张图太大没有 budget 命中 → 那张走 eager。**`try_encode` 返回 None 时自动降级 eager**，保证不报错。
3. **纯 eager**：CG 未启用或全失败。

> **"回退到 eager"是正确性底线**：CUDA Graph 只是性能优化，绝不能因为 CG 失败而让推理失败。`try_encode` 返回 `None`（miss）正是这个降级信号。

---

## 6. 与 Decode CUDA Graph 的对比

| 维度 | Decode CG（v3） | ViT CG（v4） |
|------|----------------|-------------|
| 变长来源 | batch 大小固定，seq 长度=1 | 图 token 数随分辨率变 |
| 预算策略 | 单一大 batch | 多档 budget |
| 静态 buffer | 一个固定大 buffer | 每档各一套 buffer |
| 元数据 | KV page 固定 | cu_seqlens / max_seqlen 固定 |
| 捕获时机 | 启动时一次 | 启动时一次（`warm_vit_cg`） |
| 回退 | 无（batch 内定） | 有（eager 兜底） |

**两个 CG 正交，可同时启用**（`bench/mm_cg_suite_v4.py` 默认两者都开）。ViT CG 加速 prefill 的视觉编码段，Decode CG 加速 token 生成段。

---

## 7. 动手实验

1. 分别用 `--no-vit-cg` 和默认（开 ViT CG）跑 `mm_cg_suite_v4.py`，对比 `vit_encode_s` 和 `ttft_s`。
2. 打印 `VisionEncoderCudaGraph.hits / misses`，观察什么尺寸的图像命中 budget、什么尺寸 miss（`mm_cg_suite_v4.py` 的 `vit_cg_hits` 字段）。
3. 改 `token_budgets` 为一档（如只 `[2048]`），观察小图的 hit 率变化和 padding 浪费。
4. 故意把 `max_batch_size` 设为 1，观察多图场景走向"贪心打包"路径。

---

## 8. 面试 / 教学提问

- 为什么 ViT 需要 CUDA Graph？eager 的瓶颈在哪？
- 变长输入怎么和"CUDA Graph 要求固定地址"调和？预算式设计的核心思想是什么？
- `plan_attn=False` + `bind` 为什么是 CG 捕获的必要条件？
- 捕获前为什么要预热一次 forward？
- `_copy_pad` 和 cu_seqlens 的 zero-length padding 在做什么？不变式是什么？
- `try_encode` 返回 None 时发生什么？为什么这是正确性底线？

---

**下一讲**：[11-编码器预算调度](11-编码器预算调度.md) —— 调度器如何保证"未编码不过图"。
