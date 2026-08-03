# Engine 推理引擎

> [engine.py 源码](../src/engine.py) | [architecture 总览](./architecture.md) | [model 模型实现](./model.md)

<span style="color:#1d4ed8">**模块名**</span> · <span style="color:#b45309">**核心机制**</span> · <span style="color:#059669">**实现细节**</span> · <span style="color:#7c3aed">**数据流**</span>

`model.py` 解决了「模型怎么算」，`engine.py` 解决「怎么拿模型跑起来、一个字一个字地吐出来」。v0 的 Engine 保持极简，把**加载、循环、采样、流式输出**收进一个类，为后续 KV Cache 和调度器升级留出清晰边界。

## 1. Engine 在系统中的位置

```
run.py
  └─ EngineConfig          读 HF config，管路径与超参
  └─ Engine.__init__       组装 tokenizer / model / sampler
  └─ Engine.generate       自回归循环，yield 每个 token 文本
       ├─ Qwen3ForCausalLM  forward + compute_logits
       └─ Sampler           logits → next_token_id
```

<span style="color:#b45309">**机制**</span> Engine 是**编排层**，不实现任何 Transformer 算子。它负责把 Python 侧的 prompt 变成 GPU 张量，驱动模型前向，再把采样结果 decode 回字符串。

<span style="color:#059669">**细节**</span> `lm_head` 不在 `model.forward` 里调用，而是在 `generate` 循环里通过 `compute_logits` 单独投影。这和 `model.py` 里的拆分设计一一对应，后续换 backbone 或做张量并行时，Engine 的循环结构不用大改。

## 2. 启动阶段 `__init__`

初始化按固定顺序走四步，每步各管一件事。

```py
class Engine:
    def __init__(self, config: EngineConfig):
        # ① tokenizer
        # ② dtype + device
        # ③ model 构建与权重加载
        # ④ sampler
```

### 2.1 Tokenizer

```py
self.tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
if self.tokenizer.pad_token is None:
    self.tokenizer.pad_token = self.tokenizer.eos_token
self.eos_token_id = self.tokenizer.eos_token_id
```

<span style="color:#059669">**细节**</span> 部分模型的 tokenizer 没有 pad_token，直接用于 batch 填充会报错。这里回退到 `eos_token`，是 HF 生态的常见写法。`eos_token_id` 缓存下来，生成循环里用来判断何时停止。

### 2.2 精度与设备

```py
self.device = torch.device("cuda:0")
self.dtype = self._resolve_dtype()
```

```py
def _resolve_dtype(self):
    if self.config.dtype != "auto":
        return getattr(torch, self.config.dtype)
    dt = self.config.hf_config.torch_dtype
    return dt if isinstance(dt, torch.dtype) else torch.float16
```

<span style="color:#b45309">**机制**</span> `dtype` 优先读用户配置，否则跟 checkpoint 里记录的 `torch_dtype` 走，再不行就落到 `float16`。

<span style="color:#059669">**细节**</span> v0 硬编码 `cuda:0`，没有多卡逻辑。

### 2.3 模型加载与编译

```py
self.model = Qwen3ForCausalLM(config.hf_config)
load_model(self.model, config.model)
self.model = self.model.to(dtype=self.dtype, device=self.device)

if not config.enforce_eager:
    self.model = torch.compile(self.model, mode="reduce-overhead")
```

<span style="color:#b45309">**机制**</span> 先建空骨架，再流式灌权重（第 3 节 `model.md`），最后 `.to()` 上设备/转 dtype。**加载顺序是刻意设计的**——若先 `.to(device)` 再灌权重，加载过程会反复在 CPU/GPU 间搬运张量，浪费显存；先建骨架、权重灌进参数后统一迁移，只搬一次。

<span style="color:#059669">**细节**</span> `enforce_eager=True` 时跳过 `torch.compile`。`run.py` 里默认开启，是为了调试时跳过编译等待和 trace 报错不好定位；关掉（`enforce_eager: false`）可让 `torch.compile(mode="reduce-overhead")` 生效——融合 kernel + CUDA Graph 消除启动开销（profile 实测 eager → compile 约 3.7x 加速）。

<span style="color:#059669">**细节**</span> 只对 `model` 做 compile，`Sampler` 不 compile。采样分支里有大量条件判断和 `multinomial`，compile 收益小，反而增加复杂度。

### 2.4 Sampler

```py
self.sampler = Sampler().to(device=self.device)
```

<span style="color:#059669">**细节**</span> `Sampler` 继承 `nn.Module` 但没有可学习参数，`.to(device)` 只是为了和模型保持同一设备，方便在 `generate` 里直接传 GPU 张量。

## 3. 生成阶段 `generate`

这是整个 v0 最核心的循环。

```py
@torch.inference_mode()
def generate(self, prompt, max_new_tokens=128, temperature=0.0, ...):
    input_ids = ...   # encode prompt
    # 采样超参 → GPU 标量张量
    for _ in range(max_new_tokens):
        ids = torch.tensor([input_ids], ...)
        positions = torch.arange(len(input_ids), ...).unsqueeze(0)
        h = self.model(ids, positions)
        logits = self.model.compute_logits(h[:, -1:]).squeeze(1)
        next_id = self.sampler(logits, ...).item()
        yield self.tokenizer.decode([next_id])
        input_ids.append(next_id)
        if next_id == self.eos_token_id or len(input_ids) >= self.context_len:
            break
```

### 3.1 Prompt 入口

<span style="color:#b45309">**机制**</span> `prompt` 支持两种形态，原始字符串或已 tokenize 的 `list[int]`。字符串走 `tokenizer.encode`，整数列表直接当 `input_ids` 用，方便单测绕过 tokenizer。

<span style="color:#059669">**细节**</span> 超长 prompt 会从左侧截断，只保留最后 `context_len` 个 token。v0 没有滑动窗口或分段摘要，截断是简单粗暴的硬切。

### 3.2 采样参数上 GPU

```py
def _t(v, dtype):
    return None if v is None else torch.tensor([v], device=self.device, dtype=dtype)

temp_t = None if temperature <= 0 else _t(temperature, torch.float)
```

<span style="color:#b45309">**机制**</span> 采样超参在循环外一次性转成 GPU 上的 `[1]` 形状张量，循环内不再反复创建。

<span style="color:#059669">**细节**</span> `temperature <= 0` 时 `temp_t` 直接设为 `None`，`Sampler` 检测到全部为 `None` 就走 `argmax` 贪心路径，省掉 temperature scaling 和 `multinomial` 的开销。这是 v0 默认 `temperature=0.0` 时推理最快的原因。

### 3.3 逐步前向

<span style="color:#7c3aed">**数据流**</span> 每一步的数据形态如下

```
input_ids (Python list, 长度 N)
    │
    ▼ torch.tensor → ids [1, N]
    ▼ arange       → positions [1, N]   值为 0, 1, …, N-1
    ▼ model.forward
    ▼ h [1, N, hidden]
    ▼ h[:, -1:]    → 只取最后一个位置
    ▼ compute_logits → logits [1, vocab]
    ▼ sampler      → next_id (int)
    ▼ decode + yield 字符串
    ▼ append 到 input_ids，N 变成 N+1
```

<span style="color:#b45309">**机制**</span> 经典**自回归**解码。每轮只采样一个 token，拼回序列，下一轮带着更长的序列再 forward 一遍。

<span style="color:#059669">**细节**</span> 只取 `h[:, -1:]` 做 logits，prompt 中间位置的 hidden state 算了但不用。在 v0 没有 KV Cache 的前提下，这是正确的因果 LM 行为，但也意味着**每一步都在重算全部历史 token 的 Attention**，复杂度 O(N²)。

<span style="color:#059669">**细节**</span> `positions` 每轮从 0 重新 `arange` 到 `len-1`，不是只传增量位置。这和 `model.py` 里「positions 从外层传入」的设计配套，下一版接 KV Cache 时，这里要改成只传新 token 的位置偏移。

### 3.4 流式输出与停止条件

```py
yield self.tokenizer.decode([next_id])
```

<span style="color:#b45309">**机制**</span> `generate` 是生成器，每产出一个 token 的文本就 `yield` 一次。`run.py` 用 `for tok in engine.generate(...)` 边收边 `print`，用户能实时看到输出，不必等整段生成完毕。

<span style="color:#059669">**细节**</span> 每次只 decode 一个 token id，不做 batch decode。对中文等多字节字符，某些 tokenizer 的一个 id 可能对应半个字符，流式打印时偶尔会出现「半个字先出来」的现象，这是 tokenizer 本身的特性，不是 Engine 的 bug。

停止有两条路

1. 采样到 `eos_token_id`
2. `len(input_ids) >= context_len`，序列顶满上下文窗口

两条都不触发时，由 `max_new_tokens` 控制最大轮数。

### 3.5 `@torch.inference_mode()`

<span style="color:#059669">**细节**</span> 比 `no_grad` 更激进：不仅不记录梯度，连 autograd 的自动求导机制整体跳过，推理场景下省开销、防误触。整个 `generate` 循环都跑在这个模式里——模型前向、logits 投影、采样全部无梯度。

### 3.6 v0 的关键特征：每步重算全序列（无 KV Cache）

<span style="color:#b45309">**机制**</span> 第 3.3 节看到 `input_ids` 每轮都重建 `[1, N]`、`positions` 每轮从 0 `arange`。这正是 v0 **没有 KV Cache** 的标志：

```
decode 第 t 步：
  输入: 完整历史 [t0, t1, ..., t_{t-1}]（长度 N = t）
  计算: 整条序列再过一次全部 28 层 Transformer
  产出: 只用最后一个位置的 logits
```

<span style="color:#059669">**细节**</span> 每一步都在**重算所有历史 token 的 Attention**，复杂度 O(N²)。虽然只有最后一个位置被采样，但前 N-1 个位置的注意力分数全都白算了。这是 v0 最大的性能瓶颈，也是下一版 v1 引入 KV Cache 后要解决的第一个问题——缓存历史 K/V，decode 每步只对新增 token 做 O(1) 计算。

<span style="color:#059669">**细节**</span> 正因为无 KV Cache，`positions` 每轮传完整 `0..N-1`，和 `model.py` 里「positions 从外层传入」的设计配套。v1 接 KV Cache 时，这里只需改成传增量位置偏移（如 `[N]`），RoPE 的 `index_select` 接口天然支持。

## 4. 与 Sampler 的协作

Engine 不关心采样数学细节，只传采样参数

```py
next_id = self.sampler(logits, temp_t, top_p_t, top_k_t).item()
```

`Sampler` 内部的策略链（temperature → top-k → top-p → softmax → multinomial）在 `layers.md` 第 8 节已有说明。Engine 侧只需记住两点

<span style="color:#059669">**细节**</span> 参数顺序是 `(logits, temperatures, top_ps, top_ks)`，和 `generate` 函数签名的 `(temperature, top_p, top_k)` 顺序不同，调用时注意对齐。

<span style="color:#059669">**细节**</span> `.item()` 把 GPU 标量拉回 Python int，用于 `input_ids.append` 和 `eos` 判断。这一步是 CPU/GPU 同步点，每生成一个 token 都会触发一次，后续做 CUDA Graph 或批量解码时需要格外关注。

## 5. v0 的刻意简化

当前 Engine 是一个**能跑通、能流式、能采样**的最小闭环，以下能力都没做

| 缺失能力 | v0 现状 | 后续版本 |
| :--- | :--- | :--- |
| KV Cache | 每步全序列重算 | 缓存历史 K/V，只算新 token |
| 批处理 | batch size 固定为 1 | Continuous Batching |
| 显存管理 | 无 Block 分配 | PagedAttention |
| 异步调度 | 同步单请求 | 多进程 + 请求队列 |
| 分布式 | 单卡 `cuda:0` | 张量并行 / 流水线并行 |

<span style="color:#b45309">**机制**</span> 先把「prompt → token → forward → sample → decode」这条链路跑通，再逐层加优化。Engine 的接口形状（`generate` 收 prompt、yield 文本）在后续版本中应尽量保持稳定，优化的重心放在 `model.forward` 内部和调度层。

## 6. 一次完整调用的路径

以 `run.py` 为例

```
EngineConfig(model="~/huggingface/Qwen3-0.6B")
    │
    ▼ Engine.__init__
    │   tokenizer ← HF
    │   model     ← Qwen3ForCausalLM + safetensors 权重
    │   sampler   ← 空 Module，上 GPU
    │
    ▼ engine.generate("Hello, my name is", max_new_tokens=400, ...)
    │   encode → [39, 9370, 11, 716, 374]
    │   loop × N
    │     forward 全序列 → 末位 logits → sample → yield " Alice"
    │     ...
    │   遇 EOS 或达到上限 → 生成器结束
    │
    ▼ run.py 统计 tok/s
```

至此，从 `run.py` 敲下回车，到 GPU 上一个字一个字地流式输出，Engine 完成了 v0 推理链路的全部编排。下一章进入 KV Cache，解决「每步重算全序列」这个最大的性能瓶颈。
