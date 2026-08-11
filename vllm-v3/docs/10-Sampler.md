# 06 · v3 Sampler 相对 v2 的性能优化

本文与 [01-v2性能瓶颈](./01-v2性能瓶颈.md) 中【瓶颈 E — 输入/采样】对应。  
v3 在 **不改调度语义** 的前提下，对 Sampler 做了六处实质性改进，逐条分析如下。

## 一、对照表

| # | 优化点 | v2 | v3 | 收益类型 |
|---|--------|-----|-----|---------|
| 1 | top-k 算法 | 全量 `sort()` O(n log n) | `topk()` O(n log k) | **计算量** |
| 2 | top-p 算法 | `sort(ascending=False)` × 1 | `sort(descending=True)` × 1 | **消除二次 sort** |
| 3 | 采样方式 | `torch.multinomial()` | Gumbel-Max（`exponential_ / argmax`） | **消除 sort** |
| 4 | greedy 快路径 | 无 | `greedy_mask.all()` → 直接 `argmax` | **免计算** |
| 5 | 温度裁剪 | `torch.where(t==0, 1.0, t)` | `clamp(min=1e-5)` + greedy 分离 | **消除分支** |
| 6 | 数值稳定性 | 无 NaN/disabled 处理 | `nan_to_num`, disabled rows, float32 | **正确性** |


## 二、逐个优化详解

### 优化 1：top-k 从全量 sort 改为 topk

这是 v2 采样中最大的性能漏洞。

**v2：**

```python
# v2 — 对全部 vocab_size 做 sort，然后 gather
sorted_logits, sorted_idx = logits.sort(dim=-1, descending=False)
cutoff = sorted_logits.gather(1, (vocab - k).unsqueeze(1))
sorted_logits = sorted_logits.masked_fill(sorted_logits < cutoff, -float("inf"))
```

- 按 **升序** 排列整个 logits 向量（vocab_size 通常 ≥ 128,000）
- 然后从尾部 `gather` 第 k 大的值作为截断阈值
- 复杂度：**O(n log n)**，n = vocab_size

**v3 实现：**

```python
# v3 — 只取最大的 max_k 个值
k_eff = torch.where(k <= 0, torch.full_like(k, vocab), k)
max_k = int(k_eff.max().item())
topk_vals, _ = torch.topk(logits, max_k, dim=-1)
# Threshold is the k-th value per row (index k_eff-1)
gather_idx = (k_eff - 1).clamp(min=0).unsqueeze(-1)
thresholds = topk_vals.gather(1, gather_idx)
mask = logits < thresholds
```

- `torch.topk(logits, max_k)` 只找最大的 max_k 个（通常 0~100，远小于 128,000）
- 复杂度：**O(n log k)**，k << n
- 然后 `gather` 取各行自己的 threshold → `masked_fill`

**量化收益：** 假设 vocab_size=128,000，启用 top_k=50：
- v2：`sort(128000)` ≈ 128000 × log₂(128000) ≈ 128000 × 17 ≈ 2,176,000 次比较
- v3：`topk(128000, 50)` ≈ 128000 × log₂(50) ≈ 128000 × 5.6 ≈ 720,000 次比较  
- **约 3 倍的比较次数缩减**。且 v3 还额外节省了一次 `scatter_` 恢复原始顺序的开销

### 优化 2：top-p 消除二次 sort

**v2 实现：**

v2 把 top-k 和 top-p 合并在 `_top_k_top_p()` 一个函数中，共用一次 `sort(descending=False)`（升序）。但这也意味着：

- 升序 sort 对 top-k 是反直觉的（需要 `vocab - k` 从尾部算）
- 且因为 top-k 的 `masked_fill` 可能已破坏 logits 结构，后续 top-p 在"已被 mask 过的 sorted 序列"上继续做 `softmax` + `cumsum`

**v3 实现（`_apply_top_p`）：**

```python
sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
probs = torch.softmax(sorted_logits, dim=-1)
cum = probs.cumsum(dim=-1)
sorted_mask = cum - probs > p.unsqueeze(-1)
sorted_mask[..., 0] = False  # always keep the top token
```

- **降序 sort**，语义自然（
- `cum - probs > p` 而非 `cumsum <= 1-p`：标准 top-p 实现，对每一个 token 检查"不含我的累计值是否已超过阈值"

v3 把 top-k 和 top-p **拆成两个独立函数**：
- top-k 用 `topk()`，top-p 用 `sort(descending=True)`，各自走最合适的算子
- top-k 先执行（用 `topk` 而非 sort），top-p 单独做一次 sort（仅当 top-k 有非零值时才会到 top-p，且只 sort 一次）

### 优化 3：采样方式从 multinomial 换成 Gumbel-Max

**v2：**

```python
probs = torch.softmax(logits, dim=-1)
return torch.multinomial(probs, num_samples=1).squeeze(-1)
```

`torch.multinomial` 内部等价于：
1. 计算 CDF（等价于 `cumsum`）
2. 生成均匀随机数
3. 在 CDF 中二分查找

第 1 步可以复用 `softmax` 的输出，但第 3 步是对 batch 中每个 token 做 O(log n) 的搜索。

**v3：**

```python
probs = torch.softmax(scaled, dim=-1)
probs = torch.nan_to_num(probs, nan=0.0)
sample = probs.div_(
    torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
).argmax(dim=-1)
```


v3 的实现是等价变形，用 `p_i / exp(1)` 取 argmax（除法等效于对数域的减法）。

**性能对比：**

| 操作 | v2 `multinomial` | v3 Gumbel-Max |
|------|------------------|---------------|
| CDF 计算 | 内部隐式 | 不计算 CDF |
| 随机数生成 | 均匀分布 | 指数分布 |
| 搜索 | 二分 O(n log n) | `argmax` O(n) |
| kernel 数量 | 1 个 fused kernel | element-wise + argmax |

在 GPU 上，element-wise `exponential_` + `div_` + `argmax` 都是高度并行的简单操作，优于 multinomial 内部可能涉及的不规则访存。在小 batch、小 vocab 场景差距不大，但在大 vocab（128K+）且 batch 较大时，`argmax` 的规则网格计算模式比 multinomial 的搜索模式更 GPU-friendly。

### 优化 4：greedy 快路径

**v2：无。** 即使所有请求 temperature=0，仍走完整的 `logits / temperature → top_k → top_p → min_p → softmax → multinomial`。

**v3：**

```python
greedy_mask = temperatures <= 1e-5

if bool(greedy_mask.all()):
    return logits.argmax(dim=-1)  # ← 直接返回，跳过一切

# ... sampling logic ...

if bool(greedy_mask.any()):
    greedy = logits.argmax(dim=-1)
    sample = torch.where(greedy_mask, greedy, sample)
```

两层优化：
1. **全 greedy**：如果 batch 中所有请求 temperature ≤ 1e-5，跳过 top-k/top-p/softmax/采样，直接 `argmax`。**省掉近 100% 的采样开销。**
2. **混合 batch**：非 greedy 请求走正常采样，greedy 请求单独 `argmax`，最后 `torch.where` 合并。**greedy 部分零额外开销。**

在实际部署中，temperature=0 是默认且最常见的配置，这个快路径在几乎所有生产流量中都是热路径。

### 优化 5：温度处理简化

**v2：**

```python
temperatures = torch.where(temperatures <= 0, 1.0, temperatures)
```

`torch.where` 在每个元素上做分支判断，等同于 `(condition ? 1.0 : temperatures)`。

**v3：**

```python
greedy_mask = temperatures <= 1e-5
safe_temp = temperatures.clamp(min=1e-5)
```

`clamp(min=1e-5)` 是无分支的向量操作，GPU 执行效率高于逐元素的 `where`。然后再配合 greedy 快路径处理 temperature=0 的情况，保持语义正确（greedy 时不用 `safe_temp`，直接 `argmax`）。





## 三、仍有改进空间
期待诸位的解决！