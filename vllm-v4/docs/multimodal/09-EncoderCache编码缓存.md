# 09 · EncoderCache：相同图片如何做到零编码

> 对应代码：`src/multimodal/vision.py`（`EncoderCache`）、`src/multimodal/hasher.py`（`hash_image_pixels`）、`src/models/qwen3_vl.py`（`_encode_visual`）
> 目标：讲清 `EncoderCache` 的 **token 预算 + 纯 LRU 驱逐**、缓存 key 为什么必须"绑定配置"，以及"缓存命中依赖调度窗口"这个关键不变量。

---

## 1. 为什么需要 EncoderCache

**相同图片被反复编码是最大的浪费。** 一张图过 ViT 可能花费几十毫秒（24 层 transformer）。如果：
- 同一个用户问"这张图里有什么"、"这张图里的人在干嘛"（同一张图、不同问题）
- 多轮对话里图片反复出现
- 同一张图出现在多个并发请求

这些场景都共享同一张图 → **ViT 输出（embeds + DeepStack）完全相同**。缓存它，第二次起直接复用，ViT 耗时归零。

**`EncoderCache` 缓存的是什么？** ViT 编码后的 **embedding 张量**（`[视觉token数, out_hidden]`）+ **DeepStack 特征**（`[ndps][视觉token数, out_hidden]`）。它**不缓存** pixel_values——那是处理器每次实时算出来的（第 02 讲）。

---

## 2. 缓存 key 必须"绑定配置"（设计要点）

缓存的 key 是 **图片像素的哈希**：

```62:67:src/multimodal/vision.py
    def get(self, key: str) -> tuple[torch.Tensor, list[torch.Tensor] | None] | None:
        e = self._cache.get(key)
        if e is None:
            return None
        self._cache.move_to_end(key)
        return e.embeds, e.deepstack
```

但 **key 的生成位置很关键**——它发生在像素张量**已经过处理器（缩放/归一化/tensor 化）之后**：

```src/multimodal/hasher.py
def hash_image_pixels(pixel_values: torch.Tensor) -> str:
    ...
```

**为什么必须 hash 像素张量而不是 hash 原始 PIL 图？**

> 因为**像素张量 = f(原始图, 处理器配置)**。如果 hash 原始图，那么：
> - 改 `max_side`（缩放尺寸）→ 像素张量变了，但 hash 没变 → **命中旧缓存，取回的是旧尺寸的 embeds → 与当前输入的 `grid_thw` 对不上 → 静默错位**。
> - 改归一化均值/方差 → 同样的错误。
>
> hash 像素张量则**天然把处理器配置编进了 key**：配置一变，像素就变，hash 就变，缓存自动失效。**"key 绑定数据语义，而不是绑定对象身份"**——这是缓存设计的第一原则。

对比 vLLM 官方：官方在 `MultiModalInput` 构造时算 `encoder_cache_key`，同样以**经过 processor 处理的输入**为哈希对象。v4 把这一点推到了极致：哈希对象就是 `pixel_values` 本身，零额外状态。

> **改动记录（第 14 讲 §3.1）**：v4 早期版本 hash 的是 CPU 侧 PIL 图，为此额外引入了 `config_fingerprint`（处理器配置指纹）拼进 key，还配了一个 CPU 侧 `_img_cache`。三者叠加使"配置→key"的绑定关系变得绕。现在统一改为 tensor hash 后，fingerprint 和 CPU 缓存被整体删除，代码短了、正确性还更强了。

---

## 3. EncoderCache 的核心设计

### 3.1 数据结构和预算

```50:60:src/multimodal/vision.py
    def __init__(self, max_tokens: int = 16384):
        self.max_tokens = max(1, int(max_tokens))
        self.used_tokens = 0
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()

    @staticmethod
    def _tokens(embeds: torch.Tensor, deepstack: list[torch.Tensor] | None) -> int:
        n = int(embeds.shape[0])
        if deepstack:
            n += sum(int(t.shape[0]) for t in deepstack)
        return n
```

**关键设计决策：预算单位是 token 数，不是 entry 数。**

为什么？因为一张 4K 图和一张缩略图，ViT 输出 token 数差几十倍（4K 图 `grid` 大 → patch 多 → embedding token 多）。如果按 entry 数限制（比如最多缓存 128 张），可能缓存 128 张 4K 图撑爆显存；按 token 数限制（`max_tokens=16384`），不管多少张，总 token 数有上限，显存可控。

**`_tokens` 还把 DeepStack 的 token 数也算进去** —— DeepStack 特征也占显存，且与主 embedding 同长。

### 3.2 纯 LRU：为什么不需要引用计数

```62:86:src/multimodal/vision.py
    def get(self, key: str) -> tuple[torch.Tensor, list[torch.Tensor] | None] | None:
        e = self._cache.get(key)
        if e is None:
            return None
        self._cache.move_to_end(key)
        return e.embeds, e.deepstack

    def put(
        self,
        key: str,
        embeds: torch.Tensor,
        deepstack: list[torch.Tensor] | None = None,
    ) -> None:
        # Store detached GPU tensors; callers must treat as read-only.
        embeds = embeds.detach()
        ds = [t.detach() for t in deepstack] if deepstack is not None else None
        tokens = self._tokens(embeds, ds)
        if key in self._cache:
            old = self._cache.pop(key)
            self.used_tokens -= old.tokens
        while self.used_tokens + tokens > self.max_tokens and self._cache:
            _, e = self._cache.popitem(last=False)
            self.used_tokens -= e.tokens
        self._cache[key] = _CacheEntry(embeds=embeds, deepstack=ds, tokens=tokens)
        self.used_tokens += tokens
```

整个驱逐逻辑只有三条规则，用 `OrderedDict` 的语义自动完成：

| 操作 | 行为 | 语义 |
|------|------|------|
| `get(key)` | 命中后 `move_to_end(key)` | 该 entry 变为"最近使用" |
| `put(key, ...)` | 超预算时 `popitem(last=False)` 循环踢**最老**的 | 最久未使用先淘汰（LRU） |
| 重复 `put` 同一 key | 先 `pop` 旧的再插新的 | 位置回到末尾，token 重新计数 |

**为什么纯 LRU 就够了？** 有一种担心：正在被 in-flight 请求"引用"的 embedding 如果被驱逐，会不会在 `_encode_visual` 里拿到已被释放/重写的内存？

不会。看使用方式：

```239:245:src/models/qwen3_vl.py
        for i, (mf, cs, ce) in enumerate(entries):
            if max(mf.start, cs) >= min(mf.start + mf.num_encoder_tokens, ce):
                continue
            hit = self._encoder_cache.get(mf.uid)
            if hit is not None:
                emb, ds = hit
                out[i] = emb if emb.device == device else emb.to(device=device, non_blocking=True)
```

`get` 拿到的是 `_CacheEntry` 里的张量**引用**，紧接着就被 `out[i] = emb` 拷贝到本次 prefill 的 `inputs_embeds` 里（`scatter_embeds`）。**使用窗口只在 `get → out[i] = emb` 这一个同步点内**，中间没有 yield、没有其他 `put` 能插进来驱逐它。即便后续它被 `put` 驱逐，驱逐只是把它从 `OrderedDict` 里移除，**张量内存仍被 `out` 引用着，不会 UAF**——PyTorch 的张量是引用计数的，缓存删除不销毁还在用的张量。

> 这是**教学上最容易混淆**的一点：C++ 裸指针才需要"驱逐前确保无引用"；PyTorch 张量由 Python/库引用计数托管，`OrderedDict` 的删除只是少了一个引用，不破坏使用中的内存。所以纯 LRU 在语义上是安全的。

---

## 4. 缓存命中依赖调度窗口（关键不变量）

**第 01/06 讲反复出现的一个判断**：

```239:246:src/models/qwen3_vl.py
        for i, (mf, cs, ce) in enumerate(entries):
            if max(mf.start, cs) >= min(mf.start + mf.num_encoder_tokens, ce):
                continue
            hit = self._encoder_cache.get(mf.uid)
```

`max(mf.start, cs) >= min(mf.start + num_encoder_tokens, ce)` 的意思是：**这张图的 token 区间与本次 prefill 窗口没有交集 → 这次不编码它，直接跳过。**

> **为什么窗口没有交集就跳过，而不是照常查缓存？** 因为如果窗口没覆盖图片，那图片的 embedding **根本不需要**放进本次的 `inputs_embeds`（第 06 讲 `s,e` 交集逻辑保证只有重叠才 copy）。跳过它 = 省一次 cache get + 不浪费编码。**注意这里的 `continue` 不会把它加进 `miss_i`，所以它也不会被编码** —— 这是"窗口感知编码"的核心。

**缓存命中正确性**：`mf.uid`（像素哈希）→ 命中说明"这张图在某个时刻被编码过且缓存仍在"。**像素不变，处理器配置不变，ViT 输出必然不变**（ViT 是确定性的），所以缓存复用是安全的。这正是第 2 节"key 绑定配置"的意义所在——配置变了 uid 就会变，不会误命中。

---

## 5. 一个完整的 get/put 流程

```text
第一次见图 cat.jpg（处理器配置 A）:
  processor: pixel_values_A = f(cat.jpg, 配置A)
  _encode_visual: uid = hash(pixel_values_A)，窗口覆盖图片
    → cache.get(uid) → miss
    → 加入 miss 列表 → 批量 ViT encode
    → cache.put(uid, emb, ds)          # 写入缓存，used_tokens 增加

第二次（新请求也用同一张图，同配置）:
  processor: pixel_values_A（完全相同）
  → cache.get(uid) → hit → move_to_end → 直接复用 emb（跳过 ViT）

改成配置 B（如 max_side 调大）后同一张图:
  processor: pixel_values_B ≠ pixel_values_A
  → uid' = hash(pixel_values_B) ≠ uid → cache.get(uid') → miss
  → 重新编码（旧缓存项 uid 仍在，但永远不会被误命中）
```

---

## 6. 从引用计数到纯 LRU：一次真实的重构（Bug 修复）

早期版本的 `EncoderCache` 用的是 **"token 预算 + 引用计数"** 混合策略：`retain(request_id)` / `release(request_id)` 标记哪些 entry 被 in-flight 请求使用，"零引用"的 entry 才允许被驱逐。

这个设计本身没错，但它暴露了一个**真实的 bug**：`release()` 在引擎的请求生命周期里**从未被调用**（搜索调用点只有定义）。后果是：
- `_refs` 永远累积，每个 entry 的引用永不归零。
- "零引用才能驱逐"退化为**永不驱逐**。
- 缓存超预算后打 `over-commit` warning，**显存被缓存撑爆**。

这是"引用计数缓存"的经典维护成本——必须在**所有**请求结束路径（完成 / 中断 / preempt）调用 `release`，漏一条就泄漏。**纯 LRU 不需要显式的生命周期管理**：`move_to_end + popitem` 自动完成驱逐，且如 §3.2 所述，PyTorch 张量语义下纯 LRU 不会破坏使用中的内存。因此 v4 现在**删掉了 `_refs`、`retain`、`release`、`can_allocate` 共 60 多行**，`EncoderCache` 收敛为 55 行。

> **教学价值**：这是一个教科书级的例子——**复杂度不是免费的**。引用计数给每个接口加了一个"必须正确配对"的隐式契约，一旦破坏就静默泄漏；而 LRU 用数据结构的固有语义取代了这份契约。判断标准：**如果你无法保证每个路径都调用 `release`，就不要用引用计数。**

---

## 7. 动手实验

1. 用 `bench/mm_cg_suite_v4.py` 的 `vit_encode_s` 字段对比：同一张图跑两次，第二次的 `vit_encode_s` 应接近 0（cache hit）。
2. 打印 `EncoderCache._cache` 的 key，连续两次传入同一张图但改 `max_side`（如 512→1024），观察 key 是否变化（验证"配置绑定"）。
3. 把 `max_tokens` 调小（如 256），观察 `put` 里的 `popitem` 驱逐日志，确认最老的 entry 先被踢出。
4. 在多图 + 多请求的 E2E 里，把 `EncoderCache` 换成 `OrderedDict` 计数，验证 `hit` 次数与请求数一致。

---

## 8. 面试 / 教学提问

- `EncoderCache` 为什么用 token 预算而不是 entry 数限制？
- 为什么缓存 key 要 hash 像素张量而不是 hash 原始 PIL 图？配置指纹方案有什么问题？
- 纯 LRU 驱逐会不会破坏正在被 in-flight 请求使用的 embedding？为什么？（提示：PyTorch 引用计数 vs 裸指针）
- `get` 里 `move_to_end` 和 `put` 里 `popitem(last=False)` 合起来实现了什么语义？
- `max(mf.start, cs) >= min(...)` 这个判断在做什么？
- 引用计数缓存为什么在这里会泄漏？`release` 漏调一条路径的后果是什么？
- 什么样的场景才真的需要引用计数而不是 LRU？（提示：缓存对象本身不可拷贝、且使用窗口跨多个异步点）

---

**下一讲**：[10-ViTCUDA-Graph](10-ViTCUDA-Graph.md) —— 变长 ViT 如何用预算式 CUDA Graph 加速。
