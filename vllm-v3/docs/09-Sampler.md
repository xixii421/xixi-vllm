# v3 Sampler 相对 v2 的性能优化

本文与 [01-v2性能瓶颈](./01-v2性能瓶颈.md) 中【瓶颈 E — 输入/采样】对应。  
v3 在 **不改调度语义** 的前提下，对 Sampler 做了多处实质性改进，逐条分析如下。

## 一、对照表

| # | 优化点 | v2 | v3 | 收益类型 |
|---|--------|-----|-----|---------|
| 1 | top-k 算法 | 全量 `sort()` O(n log n) | `topk()` O(n log k) | **计算量** |
| 2 | top-p 算法 | `sort(ascending=False)` × 1 | `sort(descending=True)` × 1 | **消除二次 sort** |
| 3 | 采样方式 | 全程 `torch.multinomial()` | 纯温度路径 Gumbel-Max（`exponential_ / argmax`）；top-k/top-p/min-p 仍 `multinomial` | **纯温度路径消除 sort/CDF** |
| 4 | greedy 快路径 | 无 | 全 greedy 直接 `argmax`、混合 batch 用 `where` 覆盖 | **免计算** |
| 5 | 温度处理 | `torch.where(t==0, 1.0, t)` | `where(greedy_mask, ones, t)` + greedy 分离 | **无分支向量化** |
| 6 | 数值稳定性 | 无 NaN/disabled 处理 | `nan_to_num`、min-p 掩码、float32 | **正确性** |

## 二、逐个优化详解

### 优化 1：top-k 从全量 sort 改为 topk

这是 v2 采样中最大的性能漏洞。

**v2：** 对全部 vocab_size 做升序 `sort`，再从尾部 gather 第 k 大的值作为阈值。复杂度 O(n log n)，n = vocab_size。

**v3：**

```python
vals, idx = torch.topk(scaled, max_top_k, dim=-1)          # 只取最大 max_k 个
col = torch.arange(max_top_k, device=logits.device)
vals = vals.masked_fill(col.unsqueeze(0) >= top_ks.unsqueeze(1), float("-inf"))
sample = self._sample_ordered(vals, idx, top_ps, min_ps, generator)
```

`torch.topk(logits, max_k)` 只找最大的 max_k 个（通常 0~100，远小于 128,000），复杂度 O(n log k)，k << n。

### 优化 2：top-p 消除二次 sort

v2 把 top-k 和 top-p 合在一个函数里共用一个升序 sort；v3 拆成两个独立函数——top-k 用 `topk`，top-p 单独用 `sort(descending=True)`：

```python
vals, idx = torch.sort(scaled, descending=True, dim=-1)
```

降序 sort 语义自然，且只在启用 top-p（无 top-k 时）才 sort 一次。

### 优化 3：纯温度路径改用 Gumbel-Max

**v2：** 全程 `torch.multinomial`（内部等价于 CDF + 均匀随机 + 二分查找）。

**v3：** 只在「纯温度」（无 top-k/top-p/min-p）路径用 Gumbel-Max：

```python
def _gumbel_sample(logits, temperatures):
    scaled = logits.float() / temperatures.unsqueeze(1)
    probs = torch.softmax(scaled, dim=-1)
    gumbel = torch.empty_like(probs).exponential_().clamp_min_(1e-10)
    return probs.div_(gumbel).argmax(dim=-1)
```

`p_i / exp(1)` 取 argmax 是 Gumbel-Max 采样（除法等效于对数域减法），不计算 CDF、不二分查找，element-wise + argmax 都是规则网格计算。**top-k/top-p/min-p 路径仍需有序化 + `multinomial`**（见下），因为 Gumbel 技巧无法直接表达“截断到 top-k/top-p 支撑集”的约束。

**为什么这里不能加 `@torch.compile`：** 连续批处理中 decode 的 batch size 随请求陆续完成而动态递减，`torch.compile` 默认按输入形状特化（`dynamic=False`），每个新 batch 都触发一次重编译（秒级），编译开销被计入稳态吞吐。实测去掉装饰器后，T=0.6 变长负载下 v3 端到端吞吐从 1001 回到 2957 tok/s（约 3 倍）。采样器每步只跑几个轻量 kernel，融合收益（几十微秒）远小于重编译代价（秒级），因此 eager 才是正确选择，与 ullm / 官方 vLLM 的采样器一致。

### 优化 4：greedy 快路径

v2 无快路径，即使 temperature=0 也走完整 softmax → 采样。

v3 两层优化：

```python
if all_greedy:
    return logits.argmax(dim=-1)          # 全 greedy：直接返回

greedy_mask = temperatures <= 1e-5 if any_greedy else None
# ... sampling logic ...
if any_greedy:
    sample = torch.where(greedy_mask, logits.argmax(dim=-1), sample)
```

1. **全 greedy**：batch 中所有请求 temperature≈0，跳过 top-k/top-p/softmax/采样，直接 `argmax`；
2. **混合 batch**：非 greedy 正常采样，greedy 单独 `argmax`，`torch.where` 合并。

`all_greedy`/`any_greedy`/`use_top_k` 等开关在 `Batch._fill_sampling`（CPU 侧打包）里一次性判定并以标量传入，**forward 全程无 GPU→CPU 同步**。

### 优化 5：温度处理向量化

**v2：** `temperatures = torch.where(temperatures <= 0, 1.0, temperatures)`。

**v3：**

```python
safe_temp = torch.where(greedy_mask, torch.ones_like(temperatures), temperatures)
```

用 `where` 把 greedy 行置为 1.0 后采样，最后再用 `where` 覆盖回 `argmax`——避免除以 0，且保持整批向量化、无逐元素 Python 分支。

### 优化 6：数值稳定性

`nan_to_num`、min-p 的 `max_p * min_p` 掩码、logits 转 float32 缩放，保证极端输入不产生 NaN/非法 token。

## 三、性能结论

- **大 batch 是采样器真正成为瓶颈的区间**：v2 在 bs=256 时对整词表全量 sort，耗时随 batch 线性放大；v3 用 `topk` 把工作集压到 20 列，后续过滤/采样在归约空间进行（实测见 [02-实测证据](./02-实测证据.md) 1.8）。
- **greedy 全程最快**：`temperature≈0` 直接 `argmax` 短路，是生产默认热路径。
- **只开 top-p 无归约收益**：没有 top-k 时 v3 与 v2 一样要整词表 sort，收益来自省去 scatter 回填与一次 softmax。
- **小 batch 略慢**：亚毫秒级采样在延迟受限区间，v3 的多分支固定开销占比偏高，属「小 batch 可忽略的亚毫秒代价换大 batch 上的 8.5x 与 greedy 21x」的取舍。

## 四、仍有改进空间

期待诸位的解决！
