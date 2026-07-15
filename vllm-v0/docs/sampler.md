# Sampler 采样器

> [sampler.py 源码](../src/sampler.py) | [engine 推理引擎](./engine.md)

<span style="color:#1d4ed8">**模块名**</span> · <span style="color:#b45309">**核心机制**</span> · <span style="color:#059669">**实现细节**</span> · <span style="color:#7c3aed">**数据流**</span>

模型前向结束后，我们手里拿到的是词表上每个 token 的**原始分数** logits，还不是概率。Sampler 的任务就是把这 15 万维的分数向量，变成**下一个 token 的 id**。它决定了生成结果是确定性的复读，还是带随机性的创作。

## 1. 采样在推理链路中的位置

```
lm_head 输出 logits [1, vocab_size]
    │
    ▼ Sampler.forward
    │   过滤 · 缩放 · 归一化 · 随机抽取
    ▼
next_token_id (int)
    │
    ▼ tokenizer.decode
生成文本的一个字/词
```

<span style="color:#b45309">**机制**</span> 自回归生成每步都要做一次「从分布中选一个 token」的决策。选法分两大类

| 模式 | 行为 | 典型场景 |
| :--- | :--- | :--- |
| **贪心 Greedy** | 永远选分数最高的 token | 代码补全、事实问答、压测吞吐 |
| **随机采样** | 按概率随机抽，可叠加多种过滤器 | 聊天、创意写作、需要多样性的任务 |

v0 的 `Sampler` 把两条路径收进同一个 `forward`，根据传入参数自动分流。

## 2. 整体设计

```py
class Sampler(nn.Module):
    def forward(self, logits, temperatures=None, min_ps=None, top_ps=None, top_ks=None):
        ...
```

<span style="color:#b45309">**机制**</span> 继承 `nn.Module` 但没有可学习参数，纯粹是一段 GPU 上的后处理逻辑。写成 Module 是为了和 `Engine` 里其他组件一样 `.to(device)`，也方便后续用 `torch.compile` 或 CUDA Graph 统一编排。

<span style="color:#059669">**细节**</span> 四个超参都是 `Tensor | None`，形状为 `[batch_size]`，支持逐请求不同采样策略。v0 的 Engine 目前 batch size 为 1，但 Sampler 本身已为批处理预留了接口。

<span style="color:#7c3aed">**形状**</span> 输入 <span style="color:#7c3aed">`logits`</span> 为 `[batch, vocab_size]`，输出为 `[batch]` 的 token id 向量。

## 3. 贪心快路径

```py
is_all_greedy = (
    temperatures is None and min_ps is None and top_ps is None and top_ks is None
)
if is_all_greedy:
    return torch.argmax(logits, dim=-1)
```

<span style="color:#b45309">**机制**</span> 当所有采样参数都是 `None` 时，直接 `argmax`，跳过 temperature、过滤、softmax、multinomial 全套流程。

<span style="color:#059669">**细节**</span> Engine 在 `temperature <= 0` 时把 `temp_t` 设为 `None`，若 `top_p` / `top_k` / `min_p` 也没传，就会命中这条快路径。这是 v0 默认配置下推理最快的原因，也是很多线上服务的默认模式。

<span style="color:#059669">**细节**</span> 快路径不做 `float32` 转换，直接在模型输出的 dtype（通常是 `float16` / `bfloat16`）上 argmax。对贪心来说精度足够，又少一次类型转换。

## 4. 完整采样链

当任一参数非 `None` 时，走完整的六步流水线

```
logits [1, 151936]
  │
  ├─ ① float32 转换
  ├─ ② Temperature scaling     logits /= T
  ├─ ③ Top-K + Top-P 过滤      砍掉低分 token → -inf
  ├─ ④ Min-P 过滤              砍掉相对低概率 token → -inf
  ├─ ⑤ Softmax                 logits → 概率分布
  └─ ⑥ multinomial             按概率随机抽 1 个
```

### 4.1 Temperature 温度缩放

```py
logits = logits.to(torch.float)
if temperatures is not None:
    temperatures = torch.where(temperatures <= 0, 1.0, temperatures)
    logits.div_(temperatures.unsqueeze(dim=1))
```

<span style="color:#b45309">**机制**</span> 除以温度 T 后再 softmax，等价于调整分布的「尖锐程度」

| T 的取值 | 效果 |
| :--- | :--- |
| T > 1 | 分布变平坦，低分 token 也有机会被采到，输出更多样 |
| T = 1 | 不改变分布形状 |
| T < 1 | 分布变尖锐，高分 token 更占优势，输出更确定 |

<span style="color:#059669">**细节**</span> 先转 `float32` 再算。词表 15 万维的 softmax 在 `float16` 下容易下溢或精度丢失，采样阶段统一用 fp32 是业界惯例。

<span style="color:#059669">**细节**</span> `temperatures <= 0` 会被替换成 `1.0` 而非报错。真正的「零温度贪心」在 Engine 层通过传 `None` 触发快路径，Sampler 这里的处理是防御性兜底。

### 4.2 Top-K 与 Top-P

```py
logits = _apply_top_k_top_p(logits, top_ps, top_ks)
```

两个过滤器合在一个函数里，**先 K 后 P**，都在 softmax 之前对 logits 做 mask。

**Top-K** 只保留分数最高的 K 个 token，其余设为 `-inf`

<span style="color:#059669">**细节**</span> `k <= 0` 时当作「不限制」，等价于 `k = vocab_size`，即 Top-K 退化为无操作。

**Top-P（Nucleus Sampling）** 按概率从高到低累加，保留累积概率达到 P 的最小 token 集合

<span style="color:#b45309">**机制**</span> 与 Top-K 的「固定个数」不同，Top-P 是**动态集合大小**。分布尖锐时可能只留几个 token，分布平坦时可能留几百个。这让采样既能砍掉长尾噪声，又不会把合法的低分 token 一刀切掉。

### 4.3 Min-P

```py
logits = _apply_min_p(logits, min_ps)
```

<span style="color:#b45309">**机制**</span> 相对阈值过滤。先 softmax 得到概率，再删掉所有 `prob < min_p × max_prob` 的 token。`max_prob` 是当前分布里的最高概率。

<span style="color:#059669">**细节**</span> Min-P 放在 Top-K / Top-P **之后**。前三步已经收窄了候选集，Min-P 再按相对概率做最后一轮清洗，去掉「虽然进了候选集但概率实在太低」的 token。

<span style="color:#059669">**细节**</span> Min-P 是较新的采样策略（相比 Top-P），在保持多样性的同时能有效抑制胡言乱语。`run.py` 示例没开 min_p，但 Engine 接口已支持。

### 4.4 Softmax + Multinomial

```py
probs = torch.softmax(logits, dim=-1, dtype=torch.float)
return torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
```

<span style="color:#b45309">**机制**</span> 过滤后的 logits 里，被 mask 的位置是 `-inf`，softmax 后概率为 0，不会被采到。`multinomial` 按剩余概率做一次加权随机抽取。

<span style="color:#059669">**细节**</span> 如果所有 logits 都是 `-inf`（极端参数组合），`multinomial` 会报错。正常参数下不会发生，但调参时把 top_k 设成 1 同时 top_p 极低就需要注意。

## 5. `_apply_top_k_top_p` 的实现

这是 Sampler 里最有技巧的一段，值得单独看。

```py
logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
# ... 在排序后的 logits 上做 mask ...
return torch.empty_like(logits_sort).scatter_(dim=-1, index=logits_idx, src=logits_sort)
```

<span style="color:#b45309">**机制**</span> 先按分数**升序**排列，在排好序的数组上做 Top-K / Top-P 的 mask，最后用 `scatter_` 把结果**还原回原始词表顺序**。这样无论过滤逻辑多复杂，输出的 logits 维度和输入一一对应。

**Top-K 的 mask 技巧**

```py
top_k_mask = logits_sort.size(1) - k.to(torch.long)
top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
logits_sort.masked_fill_(logits_sort < top_k_mask, -float("inf"))
```

升序排列后，倒数第 K 个位置的值就是「第 K 大的分数」。所有比它小的都 mask 掉，恰好保留 Top-K。

**Top-P 的 mask 技巧**

```py
probs_sort = logits_sort.softmax(dim=-1)
probs_sum = probs_sort.cumsum(dim=-1)
top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
top_p_mask[:, -1] = False
logits_sort.masked_fill_(top_p_mask, -float("inf"))
```

在升序概率上累加，mask 掉累积概率还没达到 `1 - p` 的低分 token。强制 `top_p_mask[:, -1] = False` 是为了**至少保留 1 个 token**，防止把整个词表都 mask 光。

<span style="color:#059669">**细节**</span> Top-P 的 softmax 是在**已排序**的 logits 上做的，和最终 forward 里的 softmax 不是同一次计算。这里只是为了确定哪些 token 属于 nucleus，不影响最终概率分布的归一化。

## 6. `_apply_min_p` 的实现

```py
probs = torch.softmax(logits, dim=-1)
top_probs, _ = probs.max(dim=-1, keepdim=True)
return logits.masked_fill_(probs < min_p.unsqueeze_(dim=1) * top_probs, -float("inf"))
```

<span style="color:#b45309">**机制**</span> 阈值是**相对于当前最高分**的动态值，而不是绝对概率门槛。max_prob 很高时门槛跟着抬高，max_prob 很低时也会适当放宽，避免把合法 token 误杀。

<span style="color:#059669">**细节**</span> 这里对 logits 做了一次 softmax 只为算 mask，被 mask 后的 logits 还会在 forward 末尾再做一次 softmax。两次 softmax 开销在词表 15 万维时不可忽视，但 v0 单请求场景下完全可接受。

## 7. 与 Engine 的参数对接

Engine 里的调用方式

```py
temp_t = None if temperature <= 0 else _t(temperature, torch.float)
top_p_t = _t(top_p, torch.float)
top_k_t = _t(top_k, torch.int)
min_p_t = _t(min_p, torch.float)

next_id = self.sampler(logits, temp_t, min_p_t, top_p_t, top_k_t).item()
```

需要注意的三点

<span style="color:#059669">**细节**</span> **参数顺序不同**。`generate` 签名是 `(temperature, top_p, top_k, min_p)`，传给 Sampler 时变成 `(temperatures, min_ps, top_ps, top_ks)`，min_p 和 top_p 的位置对调了，写调用代码时要对齐。

<span style="color:#059669">**细节**</span> **None 的含义**。`None` 表示「这一步不过滤」，不是「参数为零」。Engine 用 `None` 跳过未设置的过滤器；Sampler 内部用 `<= 0` 做退化处理（如 k≤0 等于不限制）。

<span style="color:#059669">**细节**</span> **`.item()` 同步点**。`multinomial` 或 `argmax` 的结果在 GPU 上，`.item()` 拉回 Python int 时触发 CPU/GPU 同步。每生成一个 token 同步一次，是 v0 延迟的来源之一。

## 8. 一组典型参数的效果

以 `run.py` 的配置为例

```py
engine.generate(prompt, max_new_tokens=400, temperature=0.9, top_p=0.95, top_k=20)
```

| 参数 | 值 | 作用 |
| :--- | :--- | :--- |
| temperature | 0.9 | 略低于 1，输出偏确定但不死板 |
| top_k | 20 | 每步只在分数最高的 20 个 token 里抽 |
| top_p | 0.95 | 在 Top-K 结果上再做 nucleus 截断 |
| min_p | 未设置 | 不做相对概率过滤 |

<span style="color:#b45309">**机制**</span> Top-K 和 Top-P **叠加使用**时，先 K 收窄候选集，再 P 动态裁剪。K=20 保证不会从 15 万个 token 里瞎抽，P=0.95 保证不会强行留满 20 个低概率 token。两者配合是聊天场景的经典组合。

若改成 `temperature=0` 且其余不传，整条采样链被跳过，直接贪心 argmax，速度最快但输出完全确定。

## 9. 一次采样的完整路径

```
logits [1, 151936]     ← lm_head 输出，某个位置的 hidden state 投影到词表
    │
    ▼ temperature=0.9
    │   logits /= 0.9，分布略尖锐
    ▼ top_k=20
    │   151936 → 20 个候选
    ▼ top_p=0.95
    │   20 → 可能更少（nucleus 截断）
    ▼ softmax
    │   候选 token 的概率之和 = 1
    ▼ multinomial
    │   按概率随机抽 1 个，比如 id=16840
    ▼ .item() → 16840
    ▼ decode → " Alice"
```

Sampler 本身不到 60 行，却承载了 LLM **「创造性」与「确定性」之间的全部旋钮**。理解它的过滤顺序、快路径分流和 mask 还原技巧，对后续做投机采样（Speculative Decoding）、批量采样优化和 CUDA Graph 集成都很有帮助。
