# 05 · MRoPE：三维位置编码

> 对应代码：`src/multimodal/_mrope.py`（位置计算）、`src/layers/mrope.py`（cos/sin 应用）
> 目标：讲透"为什么文本是 1D 位置、图像是 3D 位置、两者如何衔接、chunked prefill 怎么续接"。

---

## 1. 问题：一张图的 token 位置该怎么算

LLM 里每个 token 的位置是一个整数（第几个 token）。但视觉 token 有**内在的三维结构**：

```text
一张图被切成 t×h×w 的 patch 网格（grid_thw）
   → t：时间维（图片恒为 1，视频是帧数）
   → h：高度方向第几行
   → w：宽度方向第几列
```

如果我们只给视觉 token 一个"在序列里的第几个"位置，模型就丢失了"上/下/左/右"的空间信息 —— 它不知道这个 patch 是图的上半部分还是下半部分。**MRoPE（Multimodal RoPE）** 就是解决这个问题：给视觉 token 一个三维位置 `(t, h, w)`，并把三维位置编码进 RoPE。

> **Qwen2-VL 引入 MRoPE**，Qwen3-VL 升级为 **Interleaved-MRoPE**（交错式，第 5.2 节），改进了 chunked 分组导致的频谱偏置。

---

## 2. 一条序列里的位置是怎么算的

`mrope_positions()` 输出 `[3, total_tokens]` 的 int64 张量，三行分别是 t / h / w 三轴的位置。

```6:25:src/multimodal/_mrope.py
def mrope_positions(
    input_ids: torch.Tensor,
    seqs: list,
    merge_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    为当前调度 batch 生成 MRoPE 所需的 3D position ids。
    返回:
        shape = [3, total_scheduled_tokens]
        三行分别对应 temporal / height / width 三个维度的位置。
    ...
    """
    total_tokens = int(input_ids.shape[0])
    pos = np.zeros((3, total_tokens), dtype=np.int64)
    off = 0
    for seq in seqs:
        ...
```

核心循环：对每个 sequence，**先按文本逻辑给所有 token 一个"线性位置"（base + arange），然后把落在图片区间内的 token 替换成三维 grid 坐标**。

```66:79:src/multimodal/_mrope.py
    for seq in seqs:
        m = int(seq.num_scheduled_tokens)

        if m <= 0:
            off += m
            continue

        base = (
            int(seq.mrope_next)
            if seq.mrope_next is not None
            else int(seq.num_cached_tokens)
        )

        pos[:, off : off + m] = base + np.arange(m, dtype=np.int64)
```

**三个轴是同一个 base** —— 文本 token 的 t/h/w 三轴位置相同（都是 `base+idx`），所以纯文本序列退化成 1D（三行相等）。这就是"MRoPE 兼容普通 RoPE"的方式。

### 2.1 图片区间内替换为 grid 坐标

```86:116:src/multimodal/_mrope.py
        for mf in seq.mm_features:
            g_tuple = getattr(mf, "grid_thw_tuple", None)

            if g_tuple is None:
                g = getattr(mf, "grid_thw", None)
                if g is None:
                    continue
                flat = g[0].tolist() if g.ndim == 2 else g.tolist()
                g_tuple = (int(flat[0]), int(flat[1]), int(flat[2]))

            t, h, w = g_tuple

            img_start = int(mf.start)
            img_end = img_start + int(mf.num_encoder_tokens)

            ov_start = max(img_start, chunk_start)
            ov_end = min(img_end, chunk_end)

            if ov_start >= ov_end:
                continue

            gh = h // merge_size
            gw = w // merge_size
            plane = gh * gw

            an = ov_end - ov_start
            vis_off = ov_start - img_start

            local = off + (ov_start - chunk_start)
            grid_part = _grid_slice(t, gh, gw, vis_off, an)
```

- `ov_start / ov_end`：**当前 chunk 与该图片区间的重叠部分**（chunked prefill 的核心：一次只处理窗口内的图）。
- `gh = h // merge_size, gw = w // merge_size`：因为 PatchMerger 把 `2×2` 合并了，所以 MRoPE 的网格是 `merge_size` 缩小后的 `(gh, gw)`。
- `grid_part = _grid_slice(...)`：生成这一段视觉 token 的三维坐标。

`_grid_slice` 把一维 index 反解成三维坐标：

```29:64:src/multimodal/_mrope.py
    def _grid_slice(
        t: int,
        gh: int,
        gw: int,
        start: int,
        length: int,
    ) -> np.ndarray:
        """
        只生成 flatten 后 [start, start + length) 这一段对应的 3D grid 坐标。

        完整 grid 的 flatten 顺序是:
            t 最慢
            h 次之
            w 最快

        对于一维 index:
            plane = gh * gw
            t_coord = index // plane
            rem     = index % plane
            h_coord = rem // gw
            w_coord = rem % gw
        """
        idx = np.arange(start, start + length, dtype=np.int64)
        plane = gh * gw
        out = np.empty((3, length), dtype=np.int64)
        out[0] = idx // plane
        rem = idx % plane
        out[1] = rem // gw
        out[2] = rem % gw
        return out
```

**flatten 顺序（t 最慢、w 最快）必须与 ViT 里 patch 展平顺序一致** —— 这是 MRoPE 与 ViT 对齐的关键约束（第 03 讲 `_rot_pos_ids` 的 reshape/transpose 也是同样的顺序）。

### 2.2 图片区间前后的衔接（origin）

图片的 3D 坐标是 **相对坐标**（0..gh-1），但整条序列是全局的。所以要把图片块"平移到"正确的位置：

```118:128:src/multimodal/_mrope.py
            if ov_start == img_start:
                origin = int(pos[0, local])
                mf.mrope_origin = origin
            else:
                if mf.mrope_origin is not None:
                    origin = int(mf.mrope_origin)
                else:
                    first_t = vis_off // plane
                    origin = int(pos[0, local]) - int(first_t)

            pos[:, local : local + an] = grid_part + origin
```

- **`origin`** 是图片块在全局序列中的"基准位置"，即图片第一块的 `(t)` 坐标。
- 图片内部的 3D 相对坐标加上 `origin`，就变成全局坐标。
- `mf.mrope_origin` 被**缓存**在 `MultiModalFeature` 上，用于 chunked prefill 跨 chunk 续接（第二次处理同一张图的后续部分时，直接用缓存的 origin）。

### 2.3 图片之后的文本位置：跳跃续接

图片后面还有文本。这些文本 token 的 t/h/w 三轴位置**要从图片块的最大坐标 +1 继续**，而不是用原来的 `base+idx`：

```132:139:src/multimodal/_mrope.py
            if ov_end == img_end and te < off + m:
                nxt = int(pos[:, local:te].max()) + 1
                tail_len = off + m - te

                pos[:, te : off + m] = nxt + np.arange(
                    tail_len,
                    dtype=np.int64,
                )
```

**为什么不能继续用 `base+idx`？** 因为图片块的坐标是三维的（比如 h 轴能到 gh-1，可能大于文本的线性位置），图片之后的文本位置必须从**图片的最大坐标**之后开始，否则位置会出现重叠/回退，破坏位置单调性。

---

## 3. chunked prefill 的增量续接：`mrope_next` / `rope_delta`

**这是 v4 相对官方实现的一个真实优化点**（官方每次全量重算），值得重点理解。

### 3.1 问题背景

一条长请求会被拆成多个 prefill chunk（`chunked prefill`）。每步只处理 `num_scheduled_tokens` 个 token。如果每步都重新算整条序列的 MRoPE，浪费极大。所以 v4 用增量状态避免重算：

- `seq.mrope_next`：**下一个 token 的 (t,h,w) 三轴统一起点**。
- `seq.rope_delta`：`mrope_next - 1 - token_idx`，decode 阶段换算位置用。

```141:146:src/multimodal/_mrope.py
        if m > 0:
            abs_end = off + m - 1
            token_idx = chunk_start + m - 1

            seq.mrope_next = int(pos[:, abs_end].max()) + 1
            seq.rope_delta = seq.mrope_next - 1 - token_idx
```

**增量逻辑**：本 chunk 处理完后，把 `mrope_next` 记为本 chunk 最后 token 三轴位置的最大值 + 1。下一个 chunk 开始时（`_mrope.py` 第 73-77 行），`base = seq.mrope_next`（若存在），**从上次结束的位置继续**，而不是从 0 或 `num_cached_tokens` 重算。

> **decode 阶段怎么用？** decode 时不再跑 `mrope_positions`，直接用 `rope_delta` 换算 1D 位置（`model_runner.py` 第 296 行 `host_max = max(host_max, len(s) + int(s.rope_delta))`）。因为 decode 的 token 都是文本，三轴位置相同，只需要知道偏移量即可扩展 cos/sin 缓存。

### 3.2 什么时候必须重置

**preempt（抢占）** 时，KV 被释放、序列要重新 prefill，MRoPE 状态也必须重置：

```210:220:src/engine/scheduler.py
    def preempt(self, seq: Sequence) -> None:
        if seq.status == SequenceStatus.FINISHED:
            return
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.kv.deallocate(seq, publish=False)
        # KV was dropped → the sequence re-prefills from scratch, so its MRoPE
        # cursor must reset too (otherwise chunked position math is stale).
        if seq.has_multimodal:
            seq.reset_mrope_state()
        self.waiting.appendleft(seq)
```

**不重置会怎样？** 序列从第 0 个 token 重新 prefill，但 `mrope_next` 还停在被抢占的位置，导致位置错乱（视觉 token 的位置、图片后文本的续接都会算错）。这是多模态 chunked prefill 的正确性关键（第 08 讲会再讲）。

---

## 4. GPU 侧：`MRotaryEmbedding` 怎么用 3D 位置做 RoPE

`_mrope.py` 算的是**位置 id**（`[3, N]` 的整数），真正做旋转的是 `layers/mrope.py` 的 `MRotaryEmbedding`。

```38:70:src/layers/mrope.py
    def _cos_sin(self, positions: torch.Tensor):
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        cos3, sin3 = self.cos_sin_cache[positions].squeeze(2).chunk(2, dim=-1)
        if self.mrope_interleaved:
            cos, sin = cos3[0].clone(), sin3[0].clone()
            for dim, off in ((1, 1), (2, 2)):
                idx = slice(off, self.mrope_section[dim] * 3, 3)
                cos[..., idx] = cos3[dim, ..., idx]
                sin[..., idx] = sin3[dim, ..., idx]
            return cos.unsqueeze(1), sin.unsqueeze(1)
        t, h, w = self.mrope_section
        cos = torch.cat([cos3[0, :, :t], cos3[1, :, t : t + h], cos3[2, :, t + h :]], -1).unsqueeze(1)
        sin = torch.cat([sin3[0, :, :t], sin3[1, :, t : t + h], sin3[2, :, t + h :]], -1).unsqueeze(1)
        return cos, sin

    def forward(self, positions, query, key):
        # 1-D → plain RoPE (decode / text); 3-D → MRoPE (MM prefill).
        if positions.ndim == 1 or positions.shape[0] < 3:
            pos = positions if positions.ndim == 1 else positions[0]
            fn = RotaryEmbedding._forward_cuda if query.is_cuda else RotaryEmbedding._forward_native
            return fn(self, pos, query, key)
        cos, sin = self._cos_sin(positions)
        if query.is_cuda:
            return _rope_cuda(
                query,
                key,
                cos.squeeze(1).contiguous(),
                sin.squeeze(1).contiguous(),
                self.head_size,
                self.rotary_dim,
            )
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)
```

### 4.1 一维 vs 三维自动切换

`forward` 里最关键的一行：

```55:59:src/layers/mrope.py
        # 1-D → plain RoPE (decode / text); 3-D → MRoPE (MM prefill).
        if positions.ndim == 1 or positions.shape[0] < 3:
            pos = positions if positions.ndim == 1 else positions[0]
            fn = RotaryEmbedding._forward_cuda if query.is_cuda else RotaryEmbedding._forward_native
            return fn(self, pos, query, key)
```

- decode 阶段传 `[N]` 的 1D 位置 → 走普通 RoPE（与纯文本 Qwen3 完全一致）。
- prefill 多模态传 `[3, N]` 的 3D 位置 → 走 MRoPE。
- **这保证了 decode 的 RoPE 与纯文本完全对齐**（零回归），是 v4 让 decode 对多模态透明的关键机制之一。

### 4.2 Interleaved vs Chunked（Qwen3 升级点）

- **Chunked MRoPE**（Qwen2.5-VL）：head_dim 被切成三段连续区间，`[0:t)` 用 t 轴，`[t:t+h)` 用 h 轴，`[t+h:…)` 用 w 轴。
- **Interleaved MRoPE**（Qwen3-VL，默认开启）：三个轴的频率**交错分布**在 head_dim 上，每个轴都覆盖全频段，避免某轴只有低频/高频（长视频推理的频谱偏置问题）。

```42:48:src/layers/mrope.py
        if self.mrope_interleaved:
            cos, sin = cos3[0].clone(), sin3[0].clone()
            for dim, off in ((1, 1), (2, 2)):
                idx = slice(off, self.mrope_section[dim] * 3, 3)
                cos[..., idx] = cos3[dim, ..., idx]
                sin[..., idx] = sin3[dim, ..., idx]
```

默认 `mrope_section = (24, 20, 20)`（head_dim=64 时 24+20+20=64），`mrope_interleaved=True`。

### 4.3 `ensure_cache_size`：动态扩表

`cos_sin_cache` 是一张 `[max_pos, rotary_dim]` 的大表（第 2 讲提过），`ensure_cache_size` 按需扩展：

```25:36:src/layers/mrope.py
    def ensure_cache_size(self, need: int) -> None:
        if need <= self.cos_sin_cache.shape[0]:
            return
        length = need + 1024
        dev = self.cos_sin_cache.device
        inv = 1.0 / (
            self._rope_base
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device=dev) / self.rotary_dim)
        )
        t = torch.arange(length, dtype=torch.float, device=dev)
        freqs = torch.einsum("i,j -> ij", t, inv)
        self.cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze_(1)
```

**为什么不预先分配很大？** MRoPE 的位置可能因为图片而"跳跃"到很大（图片块坐标 + 文本位置），预先分配 `max_position_embeddings` 全表浪费显存。用增量扩展（`need + 1024` 留余量），只在真实需要时扩。

---

## 5. 一个具体例子

假设一条序列：`[text_0..text_4] [image: 2×2 grid] [text_5..text_7]`，`merge_size=2`，图片 grid_thw=`(1,4,4)`（`gh=gw=2`）。

```text
token 位置:   0  1  2  3  4  |  5  6  7  8  |  9  10 11
              图开始于 token 5，占 4 个视觉 token（2×2 网格）:
                token5: (t=0, h=0, w=0)
                token6: (t=0, h=0, w=1)
                token7: (t=0, h=1, w=0)
                token8: (t=0, h=1, w=1)

文本部分三轴 = (idx, idx, idx)：
  0..4:  (0,0,0)..(4,4,4)
图片部分：origin = 5（token5 原本的 t 坐标）
  token5: (5+0, 5+0, 5+0) = (5,5,5)
  token6: (5+0, 5+0, 5+1) = (5,5,6)
  token7: (5+0, 5+1, 5+0) = (5,6,5)
  token8: (5+0, 5+1, 5+1) = (5,6,6)
图片后文本：从 max(5,5,5,5,6,5,6,6)+1 = 7 继续
  9..11: (7,7,7) (8,8,8) (9,9,9)   ← 注意不是 (9,9,9) 而是从 7 起跳
```

**关键观察**：图片后的文本 token 9 的 t 轴位置是 **7** 而不是 9（因为图片内部的坐标最大到 6，图片后要从 7 续）。这正是"图片后文本位置 = 图片最大坐标 + 1"的体现。**这也意味着 MRoPE 的"第几个 token"和 RoPE 的"第几个 token"在图片之后是脱钩的** —— 这就是为什么 decode 阶段要用 `rope_delta` 来对齐。

---

## 6. 动手实验

1. 单图 + 长文本：打印 `mrope_positions` 的 `[3, N]` 输出，画出三个轴的变化，验证"图片区间三轴不同、图片后从 max+1 续"。
2. 验证 `origin` 缓存的跨 chunk 一致性：把长图拆成两个 chunk 处理，对比与整图一次处理的 MRoPE 输出。
3. 注释掉 `preempt()` 里的 `reset_mrope_state()`，构造一个被抢占的多模态请求，观察输出错乱。
4. 对比 `mrope_interleaved=True/False` 两种模式下的 cos/sin 张量差异（head_dim 上的分布）。

---

## 7. 面试 / 教学提问

- 为什么视觉 token 需要三维位置？一维位置会丢失什么？
- 图片之后的文本 token 位置为什么要从图片最大坐标 +1 续接？
- `mrope_next` 和 `rope_delta` 分别解决什么问题？
- 为什么 decode 阶段用 1D 位置 + `rope_delta` 而不是继续跑 `mrope_positions`？
- Interleaved MRoPE 和 Chunked MRoPE 的区别和动机是什么？
- 为什么 preempt 时必须 `reset_mrope_state`？

---

**下一讲**：[06-DeepStack与特征融合](06-DeepStack与特征融合.md) —— 视觉特征怎么注入 LLM 中间层。
