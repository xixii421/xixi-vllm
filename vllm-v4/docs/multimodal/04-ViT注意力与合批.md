# 04 · ViT 注意力与合批：varlen FlashAttention 一次算完多张图

> 对应代码：`src/layers/vision/vit_attn.py`、`src/multimodal/vit.py`（`prepare_metadata`）
> 目标：讲清 ViT 注意力为什么要用 varlen（变长）FlashAttention，以及 `cu_seqlens`/`offsets`/`seqlens` 这些元数据怎么算、怎么用。

---

## 1. 为什么 ViT attention 不能简单调用

LLM 的 attention 有 **paged KV Cache**（KV 分布在不连续内存页），所以需要 `FlashInfer` 这类带 page table 的算子。但 **ViT 的 attention 不需要 KV Cache** —— 它一次前向算完，Q/K/V 都是连续张量，且所有 patch 相互可见（**双向 attention，causal=False**）。

所以 ViT attention 用的是 **`flash_attn` 的 `flash_attn_varlen_func`**（FlashAttention 社区包，不是 FlashInfer）。

```41:54:src/layers/vision/vit_attn.py
    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        assert self._cu_seqlens is not None
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=self._cu_seqlens,
            cu_seqlens_k=self._cu_seqlens,
            max_seqlen_q=self._max_seqlen,
            max_seqlen_k=self._max_seqlen,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=False,
        )
```

**为什么用 varlen？** 一个 batch 里多张图的 patch 数不同（不同分辨率 → 不同 token 数）。varlen 版本允许把多张图的 Q/K/V 拼成一个 `[total_patches, H, D]` 的大张量，用 `cu_seqlens` 告诉内核每张图的边界，**一次 kernel 调用完成所有图的 attention**，避免逐张循环（逐张循环会放大 kernel launch 开销，且无法利用 batch）。

---

## 2. `ViTAttentionBackend`：plan / bind / run 三阶段

```9:54:src/layers/vision/vit_attn.py
class ViTAttentionBackend:
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        **_kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.scale = head_dim**-0.5
        self._cu_seqlens: Optional[torch.Tensor] = None
        self._max_seqlen: int = 0

    def plan(self, cu_seqlens_cpu: torch.Tensor, seqlens: Optional[List[int]] = None) -> None:
        self._cu_seqlens = cu_seqlens_cpu.to(
            dtype=torch.int32, device=self.device, non_blocking=True
        )
        if seqlens:
            self._max_seqlen = max(seqlens) if seqlens else 0
        else:
            self._max_seqlen = int((cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item())

    def bind(self, cu_seqlens_gpu: torch.Tensor, max_seqlen: int) -> None:
        assert cu_seqlens_gpu.device.type == "cuda"
        assert cu_seqlens_gpu.dtype == torch.int32
        self._cu_seqlens = cu_seqlens_gpu
        self._max_seqlen = int(max_seqlen)

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        assert self._cu_seqlens is not None
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=self._cu_seqlens,
            cu_seqlens_k=self._cu_seqlens,
            max_seqlen_q=self._max_seqlen,
            max_seqlen_k=self._max_seqlen,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=False,
        )
```

**这个三阶段设计是第 10 讲 ViT CUDA Graph 的核心伏笔**，逐段讲：

### 2.1 `plan`：从 CPU 拷贝 cu_seqlens 到 GPU

- 输入是 CPU 上的 `cu_seqlens_cpu` 和可选的 `seqlens`。
- `_cu_seqlens` 变成 GPU 上 `int32` 的张量（`non_blocking=True`）。
- `max_seqlen` 取各图 patch 数的最大值（没给 `seqlens` 时由 `cu_seqlens` 差分得到）。

**这是 eager 路径**：每次 forward 都从 CPU 现算的 `cu_seqlens` 拷到 GPU。

### 2.2 `bind`：直接绑定一个已存在的 GPU cu_seqlens 缓冲

- 调用方（CUDA Graph 捕获方）把**预先在 GPU 上分配好的静态 `cu_seqlens` buffer** 传进来。
- **没有拷贝**，只是把指针存下来。CUDA Graph 捕获时 GPU 地址必须是固定的，所以 `bind` 相当于把"动态 H2D 拷贝"从图里挪走。

### 2.3 `run`：执行

- Q/K/V 是 `[total_patches, num_heads, head_dim]`（`Qwen3VisionAttention` 里 `view(n, num_heads, head_dim)` 之后）。
- 用 `flash_attn_varlen_func`，`causal=False`（双向），`softmax_scale = head_dim**-0.5`。

> **为什么 eager 用 `plan`、CG 用 `bind`？** CUDA Graph 捕获期间**不允许发生任何隐式的内存分配或 H2D 拷贝**（会打断捕获）。`plan` 里有 `cu_seqlens_cpu.to(device)`，这是捕获中不允许的操作。所以 CG 路径必须在捕获**之前**用 `bind` 绑定静态 buffer，捕获时只执行纯 GPU 计算。这是第 10 讲的核心机制，先在这里留个印象。

---

## 3. 元数据：`cu_seqlens` / `seqlens` / `offsets` 怎么算

`prepare_metadata` 负责把所有元数据一次算好（CPU 上算，返回 CPU+GPU 两个版本）：

```292:361:src/multimodal/vit.py
    def prepare_metadata(
        self,
        grid_thw,
        *,
        max_num_seqs: int | None = None,
        max_seqlen_override: int | None = None,
        max_patches: int | None = None,
    ):
        """Build encoder metadata.
        ...
        """
        grids = _normalize_grids(grid_thw)
        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype
        pos = torch.cat(
            [
                _bilinear(...)
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
```

### 3.1 一图一例

假设一个 batch 里有 3 张图，`grid_thw = [[1, 14, 14], [1, 21, 14], [1, 7, 7]]`：

```text
patches = [196, 294, 49]        # h*w
cu = [0, 196, 196+294, 196+294+49]
   = [0, 196, 490, 539]
seqlens = [196, 294, 49]        # 每张图 patch 数
offsets = [0, 196, 490]         # 每张图的起始位置
max_seqlen = 294
```

- **`cu_seqlens`**：`[0, 196, 490, 539]`，varlen FlashAttention 用它划分图边界。
- **`seqlens`**：`[196, 294, 49]`，`flash_attn` 需要 max_seqlen 来内部分配。
- **`offsets`**：每张图的起始 offset，eager 切分输出时用。

### 3.2 三个 padding 参数：为 CUDA Graph 准备

`max_num_seqs` / `max_seqlen_override` / `max_patches` 三个参数 **eager 路径不会传**，是 CG 专用：

- **`max_num_seqs`**：把 `cu_seqlens` 用 `cu[-1]` 填充到固定条数。多出的"空图"表现为 zero-length seq（`seqlens` 为 0），varlen FA 对 zero-length 是安全跳过。
- **`max_seqlen_override`**：把 `max_seqlen` 固定为捕获时的值，保证 `flash_attn` 内部 workspace 大小不变。
- **`max_patches`**：把 `pos/cos/sin` 用 0 填充到固定长度，保证 CG buffer 大小固定。

> **为什么是零填充而不是重复？** 静态 buffer 里超出实际 `cu_seqlens[-1]` 的 token 永远不会被 attention 读到（varlen FA 只处理到最后一个 offset）。所以填充什么数值其实都行，填 0 最简单。**关键不变式：`cu_seqlens` 的最后一个值必须 ≤ 静态 buffer 的 patch 数**，这样 FA 不会越界。

---

## 4. 多图合批 vs 逐图循环：为什么 varlen 快

**逐图循环**（v5 的做法）：
```python
outs = [flash_attn(q_i, k_i, v_i, causal=False) for each image]  # 多次 launch
```

**v4 的 varlen 合批**：
```python
q = torch.cat([q_0, q_1, q_2])   # [539, H, D]
flash_attn_varlen_func(q, k, v, cu_seqlens=[0,196,490,539], max_seqlen=294, causal=False)
```

优势：
1. **一次 launch**：GPU 计算任务合并，kernel launch 开销（CPU 侧）从 N 次降到 1 次。
2. **更大的计算块**：FlashAttention 对更大的 batch 更高效（block 调度更满）。
3. **为 CG 铺路**：固定 `cu_seqlens` 长度 + 固定 `max_seqlen`，是预算式 CUDA Graph（第 10 讲）能捕获的前提。

**代价**：多张图拼一起后 `max_seqlen` 取最大，内核对所有行都按 `max_seqlen` 分配 tile，短图有少量填充浪费 —— 这是可接受的（第 10 讲会讲怎么用预算控制这个浪费）。

---

## 5. `_backend`：后端按 dtype 缓存

```204:210:src/multimodal/vit.py
    def _backend(self, dtype: torch.dtype) -> ViTAttentionBackend:
        hd = self.hidden_size // self.num_heads
        if self._attn is None or self._attn.dtype != dtype or self._attn.head_dim != hd:
            self._attn = ViTAttentionBackend(
                self.num_heads, hd, self.pos_embed.weight.device, dtype
            )
        return self._attn
```

后端是**按 `(dtype, head_dim)` 缓存的单例**。dtype 变了（比如 fp16/bf16 切换）就重建。这样整个 ViT forward 里，`_backend(dtype)` 只会真正构造一次。

---

## 6. 动手实验

1. 打印多图（2-3 张不同分辨率）时 `prepare_metadata` 的 `cu_seqlens`，手算 `seqlens` 和 `offsets` 验证。
2. 把 `flash_attn_varlen_func` 换成 per-image 的 `flash_attn_func` 循环，用 `bench/mm_cg_suite_v4.py` 的多图场景对比 TTFT。
3. 去掉 `causal=False`（改成 `causal=True`），观察输出是否变化、验证 ViT 是双向 attention。
4. 试试 `max_num_seqs` 传一个比实际图数大的值，确认 zero-length seq 不会导致错误输出（验证 varlen FA 对空 seq 的安全处理）。

---

## 7. 面试 / 教学提问

- ViT attention 和 LLM attention 的三个区别是什么？（无 KV Cache / 双向 / 连续内存）
- `cu_seqlens` 的 `[0, 196, 490, 539]` 怎么解读？
- `plan` 和 `bind` 的区别是什么？为什么 CG 必须用 `bind`？
- 为什么 CUDA Graph 捕获期间不允许 `to(device)` 拷贝？
- `max_seqlen_override` 和 `max_patches` 各自保证什么不变？

---

**下一讲**：[05-MRoPE三维位置编码](05-MRoPE三维位置编码.md) —— 文本 1D、图像 3D、它们怎么衔接。
