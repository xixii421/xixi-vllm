# 概述

vllm-v0 是一个从零实现的 Qwen3 自回归引擎，涵盖配置加载、读取权重、搭建模型、执行前向、采样 token、流式输出等核心内容。

v0 版本虽然说是引擎，但是实际上更像是从零搭建一个大模型执行框架，只需要能够完成最基本的推理即可。

通过 v0，读者可以看到一条完整的链路：
- 权重如何从磁盘加载到内存；
- 大模型的每一层是如何被组织起来的
- 输入文本是如何被tokenizer 转化成为 token id的；
- 模型如何根据当前 token 预测下一个 token；
- 生成结果如何逐步流式返回。




如果只是让模型生成 token，直接用 `transformers`包就已经足够了：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", torch_dtype="auto"
).cuda()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

input_ids = tokenizer("Hello, my name is", return_tensors="pt")["input_ids"].cuda()

for _ in range(20):
    logits = model(input_ids).logits                        # 整条序列完整前向
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True) # 只取最后一个位置
    input_ids = torch.cat([input_ids, next_id], dim=-1)     # 拼回输入
    print(tokenizer.decode(next_id[0]), end="", flush=True)
```

这段代码的逻辑非常简单，就是先加载一个大模型，然后把当前的所有prompt 输入到模型中，并且取出最后一个位置的logits，选出下一个token并且再把它拼回输入中，继续重复预测。

这里的 `transformers` 是一套模型调用库，具有负责加载权重、构建模型、调用 tokenizer，并提供 generate 接口等功能。

尽管它十分方便，但是方便并不等于高效。

对我们而言，这实际上是一个黑盒，而且上面的代码每生成一个 token，都会把完整输入重新送入模型。序列越长，重复计算越多。

实际上我们可以通过自行搭建模型，来掌握大模型推理的各项细节，进而逐步优化，使其更加高效。

# 为什么需要推理框架？
我们究竟为什么需要推理框架呢？原生的模型为什么不能做推理？

1. **计算冗余极大**。如果每次生成新的Token都把包含历史所有的Token重新进行一次完整的Attention计算，那么时间复杂度就为`O(N*2)`.对于128K甚至1M的长上下文，这种重复计算会导致生成速度极慢。
2. **显存碎片化**。为了解决第一个问题，业界引入了KV Cache，用来缓存历史 Token 的 Key 和 Value。但是这又带来了更加严重的问题。传统的 HuggingFace 的 transformers 在请求到来时，会按照请求的最大可能长度一次性静态预先分配连续的显存空间。由于无法提前知道用户是输入10个字还是1000个字，因此会提前按照最大预先分配，结果会有大量的显存被闲置，同时，长短不一的显存块分配和释放会产生大量外部碎片。研究表明，原生框架中只有 20%~40% 的显存真正用于存储 KV Cache，剩余全部被浪费。 这导致单卡能同时处理的并发请求数极低。
3. **静态批处理**。为了提高吞吐量，我们需要将多个用户的请求拼成一个 Batch 丢进 GPU。传统的批处理需要等待同一个批次里最长的那个句子生成完毕，才能整体返回并且进入下一批请求，提前生成完的短句子只能进行等待，造成了巨大的算力闲置。

vLLM之所以能够成为当前LLM推理的标杆，核心就在于针对上述三个痛点，它引入了类似于操作系统级别的内存管理和细粒度的调度算法。

1. **PagedAttention**。这是vLLM的灵魂，灵感完全来源于操作系统的虚拟内存与分页管理。vLLM 不再为每个请求分配一块连续的巨大显存，而是将显存划分为固定大小的“块（Block）”。请求的逻辑 KV Cache 被映射到物理上不连续的 Block 中。只有当真正生成新的 Token 并填满当前 Block 时，系统才会去动态申请下一个 Block。这意味着在相同的 GPU 上，vLLM 可以把 Batch Size 撑大好几倍，从而极大地提升了系统的总吞吐量。此外，它还允许不同的请求共享前置 Prompt 的 KV Cache。
2. **Continuous Batching**。这是一种迭代调度，简单来说在 GPU 每次前向传播生成一个 Token 后，调度器会立刻检查：如果有请求遇到了 <EOS>（生成完毕），就立刻把它从 Batch 中踢出，腾出位置；如果有新请求在排队，就立刻把它塞进当前的 Batch 中，参与下一次的 Token 生成。这样 GPU 永远处于高负载的“饱和工作”状态，不存在因为长短句造成的算力浪费。
3. **分布式与高性能算子**。真实的模型动辄 70B、405B，单张 GPU 根本塞不下，且原生 PyTorch 算子开销过大。因此需要多种并行策略：张量并行（Tensor Parallelism）、FlashAttention集成、与CUDA Graph加速。
4. **异步多进程架构**。实际上接收请求、维护调度队列和GPU进程，这些都是需要解耦合的，因此完全可以使用异步架构。

而我们如果一上来就学习这么复杂的内容，会直接让我们淹没在巨大的学习成本中，接下来我就带大家一步一步实现每一个版本，并且进行性能和功能完整的测试。


# 自回归循环

大模型的核心是自回归生成。即模型推理预测下一个 token，反复执行，直到生成结束。

## 1. 最简循环

```python
# engine.py 的核心
for _ in range(max_new_tokens):
    # 1. 构造输入：整个序列（prompt + 已生成 token）
    ids = torch.tensor([input_ids], device="cuda", dtype=torch.long)

    # 2. 位置：[0, 1, 2, ..., current_len-1]
    positions = torch.arange(len(input_ids), device="cuda").unsqueeze(0)

    # 3. 完整 forward
    h = self.model(ids, positions)

    # 4. 取最后一个位置的 hidden state，投影到词表，采样
    logits = self.lm_head(h[:, -1:]).squeeze(1)
    next_token = self.sampler(logits, ...).item()

    # 5. 输出新 token，拼回序列，进入下一轮
    yield self.tokenizer.decode([next_token])
    input_ids.append(next_token)
```

每生成一个新 token，都要把整个序列重新跑一遍完整 forward。

![v0 自回归数据流](./posts/流程图.png)

## 2. 每一步发生了什么

第 t 步时，输入长度 N = t。

```text
input_ids [t0, t1, ..., t_{t-1}]      长度 N
    ↓
positions [0, 1, ..., N-1]            完整位置
    ↓
model 前向                             28 层全算
    ↓
h[:, -1:]                             只取最后一个位置
    ↓
lm_head → logits [1, vocab]
    ↓
sampler → next_token_id
```

## 3. 代价

每步都在重算所有历史 token 的注意力，复杂度 O(N²)。虽然只有最后一个位置被采样，前面 N-1 个位置的注意力分数全白算了。

这是 v0 最大的性能瓶颈，也是下一版 v1 引入 KV Cache 后要解决的第一个问题。缓存历史 K/V，decode 每步只对新增 token 做计算。

