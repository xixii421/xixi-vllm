结论先说：

> 你这个 `EncoderCache + Qwen3VisionTower` 模块本身是一个很轻量的“视觉 encoder 输出缓存 + ViT 包装器”。  
> 它能工作，但如果目标是接近 vLLM 官方性能/语义，那么存在比较明显的性能和工程缺陷。  
> 主要问题不是单个算子慢，而是：**缓存不是内存感知/引用计数的**、**DeepStack 状态管理不安全**、**VisionTower 缺少批处理/CUDA Graph/官方 encoder cache 集成**。

下面分开说。

---

# 一、这个模块当前做了什么？

---

## 1. `EncoderCache`

你的实现是一个简单 LRU：

```python
self._cache: OrderedDict[str, tuple[torch.Tensor, list[torch.Tensor] | None]]
```

它按字符串 key 缓存：

```python
(embeds, deepstack)
```

其中：

- `embeds` 是视觉 encoder 主输出；
- `deepstack` 是 Qwen3-VL 的多层视觉特征列表；
- 所有 tensor 都 `detach()` 后继续放在 GPU 上。

这个思路是对的：  
对于相同图片/视频，避免重复跑 ViT。

vLLM 官方的 encoder cache 也是类似目标：缓存多模态 encoder 输出，避免相同多模态输入重复编码 [[1]]。  
LMCache 文档也说明，encoder cache 按 `mm_hash` 存储 encoder 输出，形状通常是 `[num_tokens, hidden_dim]` [[3]]。

---

## 2. `Qwen3VisionTower`

这是一个薄封装：

```python
self.vit = Qwen3VisionTransformer(vision_config)
```

它提供：

```python
prepare_metadata()
forward()
deepstack_features
load_weights()
```

本质上只是把底层 `Qwen3VisionTransformer` 暴露出去。

这个封装本身没有太多逻辑，所以很多性能问题不在这一层本身，而在于它没有和 vLLM 的 encoder cache / scheduler / CUDA Graph 机制对齐。

---

# 二、性能缺陷分析

---

## 1. `EncoderCache` 只按 entry 数量 LRU，不按显存/token 数量预算

你写的是：

```python
self.max_size = max_size
...
while len(self._cache) > self.max_size:
    self._cache.popitem(last=False)
```

也就是说：

```text
max_size = 256
```

表示最多缓存 256 个 entry。

但问题是，每个 entry 的显存占用可能差异巨大。

例如：

```text
图片 A: 256 tokens * hidden_size
图片 B: 8192 tokens * hidden_size
视频 C: 32768 tokens * hidden_size
```

它们在你的 cache 里都只算 1 个 entry。

这会导致两个问题：

### 问题一：高分辨率图片/视频容易 OOM

如果 256 个 entry 都是高分辨率图像或长视频，GPU 显存可能直接爆掉。

### 问题二：小图场景浪费显存

如果所有图片都很小，256 个 entry 可能只占很少显存，但你不再继续缓存更多。

---

### 官方怎么做的？

vLLM 官方 `EncoderCacheManager` 的 `cache_size` 不是简单“多少个请求”或“多少张图片”，而是按 encoder embeddings 数量来度量 [[1]]。

也就是说，官方更关心：

```text
缓存了多少 multimodal embedding tokens
```

而不是简单缓存了多少个 Python dict entry。

因此你的 cache 应该至少改成按 token 数或 byte 数预算。

例如：

```python
def entry_size(embeds, deepstack):
    size = embeds.shape[0]
    if deepstack is not None:
        size += sum(t.shape[0] for t in deepstack)
    return size
```

或者更直接按 bytes：

```python
def entry_bytes(embeds, deepstack):
    size = embeds.numel() * embeds.element_size()
    if deepstack is not None:
        for t in deepstack:
            size += t.numel() * t.element_size()
    return size
```

---

## 2. 没有引用计数，可能驱逐正在被请求使用的 encoder 输出

你的 LRU 只根据最近访问时间驱逐：

```python
while len(self._cache) > self.max_size:
    self._cache.popitem(last=False)
```

它不知道某个 cache entry 是否还被 in-flight request 使用。

这在 vLLM 这种 continuous batching 系统里比较危险。

例如：

1. request A 开始 prefill，图片编码结果放入 cache；
2. request A 还没跑完；
3. 新请求大量进入，cache 满；
4. 你的 LRU 把 request A 的图片 embedding 驱逐了；
5. 后续如果某个阶段还需要按 key 取这个 embedding，就会 miss，导致重算或逻辑不一致。

---

### 官方怎么做的？

官方 `EncoderCacheManager` 会维护每个 `mm_hash` 当前被哪些 request 引用 [[1]]。  
它的驱逐策略是：在分配空间时，如果没有空闲空间，会优先驱逐最旧的、且没有被任何请求引用的 embedding [[1]]。

也就是说，官方不是简单 LRU，而是：

```text
LRU + reference count + allocation-time eviction
```

你的实现缺少：

```python
allocate()
can_allocate()
free()
free_encoder_input()
```

这一类生命周期管理接口 [[1]]。

---

## 3. `EncoderCache` 没有区分“可释放”和“不可释放”

官方文档里有两个概念：

```text
num_free_slots
num_freeable_slots
```

前者是当前真正空闲容量；  
后者是可以通过驱逐零引用 entry 立即回收的容量 [[1]]。

你的 cache 没有这个概念。

这会导致调度器无法判断：

```text
当前能不能安全接收一个新的多模态请求？
```

在 vLLM 里，这很重要，因为 scheduler 需要在请求进入前判断 encoder cache 是否能分配成功。

---

## 4. DeepStack 缓存方式和官方 Qwen3-VL 集成方式可能不一致

你的 cache 存：

```python
(embeds, deepstack)
```

其中 `deepstack` 是：

```python
list[torch.Tensor] | None
```

这看起来合理，但 Qwen3-VL 在 vLLM 官方实现里，DeepStack 不是简单“附加列表”这么随意。

官方 Qwen3-VL 模型里有专门的 DeepStack 处理逻辑：

```python
_compute_deepstack_embeds()
_get_deepstack_input_embeds()
_clear_deepstack_input_embeds()
```

它会把视觉塔输出拆成：

```python
multimodal_embeddings_main
multimodal_embeddings_multiscale
```

然后构造：

```python
deepstack_input_embeds
```

再传给 language model [[11]]。

另外，Hugging Face 讨论里也提到，对于 Qwen3-VL，vLLM 侧的 `image_embeds` 需要同时包含 base image embedding 和 DeepStack features [[13]]。

这意味着：

> 如果你要接入 vLLM 的 encoder cache，最好明确你的 cache 存的到底是“原始 ViT 输出列表”，还是“已经按 vLLM Qwen3-VL 格式整理好的 embedding”。

你现在这个：

```python
(embeds, deepstack list)
```

比较像内部中间状态，不是官方 encoder cache 的标准 tensor 形态。

---

## 5. DeepStack 通过 `self.vit.deepstack_features` 暴露，状态不安全

你的代码：

```python
@property
def deepstack_features(self):
    return self.vit.deepstack_features if self.vit else None
```

这依赖底层 ViT 在 forward 后把 DeepStack 特征保存在模块状态里：

```python
self._deepstack = deepstack or None
```

这个设计在单请求、单线程、立即消费的场景下可以工作。

但在 vLLM 里有几个风险：

### 风险一：batch 中多个图片/视频时状态混淆

如果一次 forward 处理多个 multimodal item，`deepstack_features` 是整批的结果，而不是某个 request 或某个 `mm_hash` 的结果。

后续如果按 request 缓存，很容易错位。

### 风险二：CUDA Graph 不友好

vLLM 官方 Qwen3-VL 支持 encoder CUDA Graph 捕获和回放 [[11]]。  
CUDA Graph 通常要求输入输出通过固定 buffer 传递，而不是依赖模块内部 Python 状态。

你这种：

```python
forward()
然后读 property
```

不太适合 CUDA Graph。

### 风险三：并发不安全

如果多个请求、多个 stream、多个 worker 交错调用，这个 property 可能读到上一次 forward 的结果。

---

### 更合理的方式

应该让 `forward()` 显式返回：

```python
embeds, deepstack
```

或者返回一个结构：

```python
@dataclass
class VisionEncoderOutput:
    embeds: torch.Tensor
    deepstack: list[torch.Tensor] | None
```

不要让调用方通过模块属性取上一次副作用结果。

---

## 6. `load_weights` 的 device/dtype 处理有性能隐患

你的代码：

```python
self.vit = Qwen3VisionTransformer(vision_config).to(
    next(iter(state_dict.values())).device
)
```

这里有两个问题。

---

### 问题一：如果 `state_dict` 为空会直接崩

```python
next(iter(state_dict.values()))
```

如果 `state_dict` 是空 dict，会抛 `StopIteration`。

---

### 问题二：只 move device，没有明确 dtype

`Qwen3VisionTransformer` 初始化时默认参数 dtype 通常是 fp32。

如果你只是：

```python
.to(device)
```

但没有：

```python
.to(dtype=bf16)
```

那么 ViT 权重可能留在 fp32。

这会带来：

1. 显存翻倍；
2. 带宽压力更大；
3. 推理速度下降；
4. 和 LLM 侧 bf16/fp16 计算不一致。

虽然后面 `load_weights()` 可能会把 checkpoint tensor copy 到 param 里，但如果 param 本身是 fp32，最终仍然是 fp32。

更合理的是：

```python
torch_dtype = next(iter(state_dict.values())).dtype

self.vit = Qwen3VisionTransformer(vision_config).to(
    device=device,
    dtype=torch_dtype,
)
```

或者外部显式传入：

```python
device=...,
dtype=torch.bfloat16,
```

---

## 7. 没有批处理多个未缓存图片/视频

你的 `Qwen3VisionTower.forward()` 是单次调用：

```python
return self.vit(pixel_values, grid_thw=grid_thw, metadata=metadata)
```

如果上层调度器每个 request 调一次，那么就会出现：

```text
request 1: vision tower forward
request 2: vision tower forward
request 3: vision tower forward
...
```

这在多并发多模态请求下会明显增加 TTFT。

---

### 官方/社区优化方向

vLLM Metal 社区明确提出过 Qwen3-VL 的优化方案：  
把多个未缓存的 Qwen3-VL 图像特征先 batch 起来，然后一次性调用 vision tower，再把编码结果切分回各个请求 [[10]]。

具体做法是：

1. 收集当前 scheduled 且未缓存的 image features；
2. concat 它们的 `pixel_values`；
3. batch 它们的 `image_grid_thw`；
4. 只调用一次 Qwen3-VL vision tower；
5. 把 hidden states 和 DeepStack outputs slice 回原始请求；
6. 保持已有 multimodal cache 行为不变 [[10]]。

这个优化对并发多模态 prefill 的 TTFT 很有帮助 [[10]]。

你的当前 wrapper 没有提供这种 batch encode 接口。

---

## 8. 没有 encoder CUDA Graph 支持

vLLM 官方 Qwen3-VL 模型实现里已经有 encoder CUDA Graph 相关接口，例如：

```python
encoder_cudagraph_forward()
prepare_encoder_cudagraph_capture_inputs()
select_encoder_cudagraph_items()
```

这些接口用于捕获和回放视觉 encoder 的 CUDA Graph [[11]]。

CUDA Graph 对 ViT 加速很重要，因为 ViT 通常有很多小 kernel：

- patch embed；
- position embedding interpolation；
- rotary embedding；
- LayerNorm；
- QKV split；
- RoPE；
- attention；
- MLP；
- patch merger；
- DeepStack merger。

如果每次都走普通 Python + CUDA launch，CPU launch overhead 会很明显。

你的模块完全没有这一层。

---

## 9. 没有保证 ViT attention 使用高效 backend

你的底层 ViT 使用：

```python
ViTAttentionBackend
```

但这一层没有体现：

- 是否使用 FlashAttention；
- 是否使用 xformers；
- 是否支持 varlen attention；
- 是否支持 Qwen3-VL 的 head_dim；
- 是否支持 CUDA Graph；
- 是否支持 fp16/bf16；
- 是否有 workspace 复用。

Qwen3-VL 在 vLLM 里曾经遇到过 flash attention head_dim 不能被 32 整除的问题 [[26]]。  
也有 issue 显示 Qwen3VL 对 upstream flash-attn 构建存在兼容性问题，需要使用 vLLM fork 或 fallback [[32]]。

所以如果你要在 vLLM 中加速，必须确认你的 attention backend 是真正高效的，而不是普通 SDPA fallback。

---

## 10. `prepare_metadata` 可能每次 forward 都重算

你的 wrapper：

```python
def prepare_metadata(self, grid_thw: torch.Tensor) -> dict:
    assert self.vit is not None
    return self.vit.prepare_metadata(grid_thw)
```

如果底层 `prepare_metadata` 每次都重新计算：

- bilinear position embedding；
- rotary cos/sin；
- cu_seqlens；
- seqlens；
- offsets；
- sizes；

那么对于相同分辨率重复请求，这部分也是浪费。

虽然它比 ViT forward 便宜，但在高并发下仍然可能成为 CPU/GPU 之间的开销点。

---

# 三、和官方版本比较

---

## 1. 官方 EncoderCacheManager 是生命周期管理器，而不是简单 LRU

你的 `EncoderCache` 是：

```text
OrderedDict + max_size + move_to_end
```

官方 `EncoderCacheManager` 则负责多模态 encoder 输出的完整生命周期 [[1]]。

它提供的能力包括：

```python
allocate()
can_allocate()
check_and_update_cache()
free()
free_encoder_input()
get_cached_input_ids()
get_freed_mm_hashes()
reset()
```

这些接口说明官方 cache 是和 scheduler 深度集成的 [[1]]。

而你的 cache 只是一个独立数据结构，不能被 scheduler 用来做资源预算。

---

## 2. 官方 cache 是 memory-aware 的

官方文档明确说，`EncoderCacheManager` 提供 memory-aware caching，用于避免相同多模态输入在不同处理阶段重复计算 [[1]]。

它的 `cache_size` 表示 cache 容量，单位是 encoder embeddings 数量 [[1]]。

你的 `max_size` 是 entry 数量，不是 embedding token 数量，也不是 byte 数量。

这是一个比较大的语义差异。

---

## 3. 官方 cache 按 `mm_hash` 共享多模态 embedding

官方 encoder cache 的核心 key 是 `mm_hash`。  
相同图片/视频/音频的多个请求可以共享 encoder 输出 [[3]]。

你的 key 是外部传入的 `str`。

如果这个 string 就是 vLLM 的 `mm_hash`，那还可以。  
但如果它只是图片 URL、请求 ID、或者不稳定的 hash，就会失去跨请求复用能力。

---

## 4. 官方 Qwen3-VL 对 DeepStack 有专门 buffer 机制

官方 Qwen3-VL 模型不是简单把 DeepStack 特征挂在模块属性上。

它会：

1. 从视觉塔输出中拆分 main embedding 和 multiscale embedding；
2. 构造 `deepstack_input_embeds`；
3. 传给 language model；
4. 用完后清理 buffer [[11]]。

这说明 DeepStack 在 vLLM 官方实现里是有明确生命周期和格式约定的。

你的：

```python
self.vit.deepstack_features
```

过于隐式。

---

## 5. 官方 Qwen3-VL 支持 encoder CUDA Graph

官方 Qwen3-VL 模型实现了 encoder CUDA Graph 相关接口，包括：

```python
encoder_cudagraph_forward()
prepare_encoder_cudagraph_capture_inputs()
select_encoder_cudagraph_items()
```

这些用于在固定 token budget、max batch size、max frames per batch 下捕获和回放视觉 encoder [[11]]。

你的模块没有这些能力。

---

## 6. 官方/社区强调 batch vision encoder 调用

vLLM Metal 的 issue 明确指出，Qwen3-VL 视觉 encoder 应该把多个未缓存图像特征 batch 后一次调用，而不是每个请求单独调用 [[10]]。

这说明在 vLLM 中，vision tower 的批处理是明确优化方向。

你的 wrapper 目前没有 batch encode 抽象。

---

# 四、如果是在 vLLM 中，如何加速这个模块？

下面按优先级给建议。

---

# 优先级 1：把 `EncoderCache` 改成官方语义：引用计数 + 内存预算

建议不要只写：

```python
max_size = 256 entries
```

而是改成：

```python
max_encoder_tokens = ...
```

或者：

```python
max_cache_bytes = ...
```

并加入引用计数。

一个简化接口可以是：

```python
class EncoderCacheEntry:
    embeds: torch.Tensor
    deepstack: list[torch.Tensor] | None
    size: int


class EncoderCache:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.used_tokens = 0
        self._cache: OrderedDict[str, EncoderCacheEntry] = OrderedDict()
        self._refs: dict[str, set[str]] = defaultdict(set)

    def can_allocate(self, key: str, size: int) -> bool:
        if key in self._cache:
            return True
        return self.used_tokens + size <= self.max_tokens

    def allocate(self, key: str, size: int):
        while self.used_tokens + size > self.max_tokens:
            self._evict_one()

    def retain(self, key: str, request_id: str):
        self._refs[key].add(request_id)

    def release(self, request_id: str):
        ...

    def _evict_one(self):
        for k in list(self._cache.keys()):
            if not self._refs.get(k):
                entry = self._cache.pop(k)
                self.used_tokens -= entry.size
                return
        raise RuntimeError("no freeable encoder cache entry")
```

这样更接近官方 `EncoderCacheManager` 的 allocate/free 语义 [[1]]。

---

# 优先级 2：DeepStack 必须显式返回，而不是通过 property

建议改成：

```python
@dataclass
class Qwen3VisionOutput:
    embeds: tuple[torch.Tensor, ...]
    deepstack: list[torch.Tensor] | None
```

然后：

```python
def forward(self, pixel_values, grid_thw=None, metadata=None) -> Qwen3VisionOutput:
    ...
```

或者如果你要按 vLLM Qwen3-VL 格式输出，可以直接返回：

```python
combined_embeds: Tensor  # [num_tokens, visual_dim + multiscale_dim]
```

因为官方 Qwen3-VL 会从合并后的 embedding 中拆分 main 和 multiscale 部分 [[11]]。  
Hugging Face 讨论也提到 Qwen3-VL 的 `image_embeds` 需要包含 base embedding 和 DeepStack features [[13]]。

---

# 优先级 3：实现 batch encode 接口

不要只支持：

```python
forward(pixel_values, grid_thw)
```

应该支持：

```python
encode_uncached_items(items: list[EncoderItem]) -> dict[str, EncoderOutput]
```

其中每个 item：

```python
@dataclass
class EncoderItem:
    mm_hash: str
    pixel_values: torch.Tensor
    grid_thw: torch.Tensor
```

然后内部：

```python
pixel_values = torch.cat([item.pixel_values for item in items], dim=0)
grid_thw = torch.cat([item.grid_thw for item in items], dim=0)

outputs = self.vit(pixel_values, grid_thw=grid_thw)

split_sizes = [...]
embeds = outputs.embeds.split(split_sizes)
deepstack = split_deepstack(outputs.deepstack, split_sizes)
```

这正是 vLLM Metal 社区提出的 Qwen3-VL 优化方式：batch 未缓存图像特征，调用一次 vision tower，再 slice 回请求 [[10]]。

---

# 优先级 4：接入 vLLM 官方 EncoderCacheManager 或模仿其接口

如果你真的在 vLLM 里跑，最好不要自己维护一个独立 cache，而是接入官方 `EncoderCacheManager`。

官方 manager 会处理：

- 是否有缓存；
- 是否能分配；
- 分配空间；
- 释放引用；
- 驱逐无引用 entry；
- 获取已释放 mm_hash [[1]]。

你的模块应该只负责：

```text
给定未缓存 pixel_values/grid_thw，产出 encoder embeddings
```

cache 生命周期交给 manager。

---

# 优先级 5：支持 encoder CUDA Graph

如果你追求高吞吐，应该给 vision tower 增加 CUDA Graph 能力。

可以参考官方 Qwen3-VL 的接口设计：

```python
prepare_encoder_cudagraph_capture_inputs()
encoder_cudagraph_forward()
select_encoder_cudagraph_items()
```

官方会基于：

```python
token_budget
max_batch_size
max_frames_per_batch
```

准备 dummy pixel_values 和 metadata，然后捕获 encoder graph [[11]]。

你可以实现一个简化版：

```python
class Qwen3VisionTower:
    def capture_encoder_graph(self, token_budget, max_batch_size, device, dtype):
        ...

    def encoder_graph_forward(self, pixel_values, metadata):
        ...
```

这对减少 CPU launch overhead 很有帮助。

---

# 优先级 6：使用高效 attention backend，并确认 Qwen3-VL head_dim 兼容

你需要确认 `ViTAttentionBackend` 是否真的使用了 FlashAttention / FlashInfer / xformers。

尤其要注意 Qwen3-VL 的 head_dim 可能触发 flash-attn 兼容性问题 [[26]]。  
如果 upstream flash-attn 不支持，需要 vLLM fork 或 fallback [[32]]。

建议检查：

```python
head_dim = hidden_size // num_heads
```

如果：

```python
head_dim % 32 != 0
```

要特别小心。

---

# 优先级 7：使用 batch-level data parallelism 加速 vision encoder

AMD ROCm 博客提到，通过 batch-level data parallelism 优化 vision encoder，可以带来最高约 45% 的性能提升 [[12]]。

这说明视觉 encoder 不一定只能跟着 LLM TP 走。

如果你的部署环境支持，可以考虑：

- vision encoder 单独 DP；
- 多个请求的图像 batch 到同一个 encoder；
- encoder 输出再 scatter 回各请求。

这对高并发多模态 TTFT 很有帮助。

---

# 优先级 8：缓存 metadata，而不是每次重算

你可以对 `prepare_metadata` 做缓存。

例如 key：

```python
metadata_key = tuple(tuple(g) for g in grid_thw.tolist())
```

缓存：

```python
pos_embeds
rotary_cos
rotary_sin
cu_seqlens_cpu
seqlens
offsets
sizes
```

但要注意：

- 如果 batch 中图片组合变化很多，cache 命中率可能不高；
- metadata 里的 tensor 不应该被 inplace 修改；
- CUDA Graph 下最好使用静态 buffer。

---

# 优先级 9：避免 cache tensor 被下游 inplace 污染

你现在：

```python
self._cache[key] = (embeds.detach(), ds)
```

`detach()` 只是断开 autograd，不复制数据。

如果下游代码做了类似：

```python
inputs_embeds[mask] = cached_embeds
cached_embeds.add_(...)
```

或者某些 fused kernel  inplace 修改了 `cached_embeds`，就会污染 cache。

建议：

1. 明确 encoder cache 输出是 read-only；
2. 下游 merge 到 `inputs_embeds` 时只读；
3. 如果不能保证，只读 copy：

```python
embeds = cached_embeds.clone()
```

但 clone 会增加带宽开销，所以最好通过工程约定避免。

---

# 优先级 10：考虑外部 encoder cache 分层

官方 encoder cache 可以通过 LMCache 等连接器扩展到 CPU、disk、remote 存储 [[3]]。

如果你 GPU 显存紧张，可以考虑：

```text
L1: GPU encoder cache
L2: CPU pinned memory cache
L3: disk / remote cache
```

对于相同图片反复请求，这能显著减少重复编码 [[3]]。

---

# 五、建议的改造方向

---

## 当前模块的问题总结

你的模块主要问题是：

1. **cache 只按 entry 数量 LRU，不按 token/byte 预算**；
2. **没有引用计数，可能驱逐正在使用的 entry**；
3. **DeepStack 通过模块属性暴露，状态不安全**；
4. **没有 batch encode 接口，容易变成每请求单独跑 ViT**；
5. **没有 encoder CUDA Graph 支持**；
6. **没有和 vLLM 官方 `EncoderCacheManager` 对齐**；
7. **load_weights 没有显式 dtype，可能 fp32 运行**；
8. **没有保证 attention backend 高效且兼容 Qwen3-VL head_dim**；
9. **metadata 可能每次重算**；
10. **cache tensor 没有只读保护**。

---

## 推荐接口设计

我建议你把模块改成这样：

```python
@dataclass
class VisionEncoderOutput:
    embeds: torch.Tensor
    deepstack: torch.Tensor | None
    sizes: list[int]


class Qwen3VisionTower(nn.Module):
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        metadata: dict | None = None,
    ) -> VisionEncoderOutput:
        ...

    def encode_batch(
        self,
        items: list[EncoderItem],
    ) -> dict[str, VisionEncoderOutput]:
        ...
```

其中：

```python
class EncoderItem:
    mm_hash: str
    pixel_values: torch.Tensor
    grid_thw: torch.Tensor
```

然后 cache 改成：

```python
class EncoderCache:
    def can_allocate(self, mm_hash: str, num_tokens: int) -> bool:
        ...

    def allocate(self, mm_hash: str, num_tokens: int) -> None:
        ...

    def get(self, mm_hash: str) -> VisionEncoderOutput | None:
        ...

    def put(self, mm_hash: str, output: VisionEncoderOutput) -> None:
        ...

    def retain(self, mm_hash: str, request_id: str) -> None:
        ...

    def release(self, request_id: str) -> None:
        ...
```

这样更接近 vLLM 官方 encoder cache 的设计 [[1]]。

---

# 六、最终判断

---

## 功能上

你的模块可以作为原型使用：

- 能缓存 encoder 输出；
- 能包装 Qwen3-VL ViT；
- 能暴露 DeepStack 特征；
- 能减少部分重复编码。

但是如果要接入 vLLM 或高并发服务，当前设计不够安全：

- DeepStack 状态不安全；
- cache 生命周期不完整；
- 缺少引用计数；
- 缺少内存预算；
- 缺少 batch/CUDA Graph。

---

## 性能上

最明显的性能缺陷是：

1. **没有 batch 多个未缓存图片/视频，导致 vision tower 可能被逐请求调用** [[10]]；
2. **没有 encoder CUDA Graph，无法消除大量小 kernel launch overhead** [[11]]；
3. **cache 不是内存/token 预算，容易 OOM 或利用率低** [[1]]；
4. **没有引用计数，调度不安全** [[1]]；
5. **DeepStack 状态管理不适合 vLLM 的 batch/CUDA Graph 场景** [[11]]。

---

## 如果只能做三件事

如果你只想优先做三个优化，我建议是：

### 第一：batch 未缓存图像/视频，一次性跑 ViT

这是最直接的吞吐和 TTFT 优化 [[10]]。

### 第二：把 cache 改成 token/byte 预算 + 引用计数

这是最接近官方 `EncoderCacheManager` 的改造 [[1]]。

### 第三：让 DeepStack 显式返回，并支持 encoder CUDA Graph

这是 Qwen3-VL 在 vLLM 中进一步加速的关键 [[11]]。