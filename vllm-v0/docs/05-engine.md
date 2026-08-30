# Engine 推理引擎

前边的文档主要负责搭建模型，而除了搭建模型之外，还需要一个模块来让这个模型真正跑起来。
我们可以选择设计一个类，来把之前提到的加载、前向推理、采样和流式输出全都放到一个类中，使其更加清晰，对外保留一个更加干净的接口。


Engine 就是这个入口。它对外提供 generate 接口，对内完成整个推理流程：把输入文本转成 token，把 token 送到模型中计算，再从模型输出的 logits 中采样下一个 token，并把生成结果逐步解码回文本。

## 1. Engine 在系统中的位置

```
run.py
  └─ EngineConfig          
  └─ Engine.__init__       
  └─ Engine.generate       
       ├─ Qwen3ForCausalLM  
       └─ Sampler          
```

EngineConfig 负责集中管理模型路径、设备、最大生成长度等配置。Engine.__init__ 根据配置创建模型并加载权重。Engine.generate 是推理入口，负责驱动整个生成过程。

Engine 本身不实现 Transformer 算子。Attention、MLP、RMSNorm、RoPE 等计算逻辑都在模型层完成。Engine 的职责是编排：准备输入，调用模型前向，调用采样器，再把生成的 token 返回给调用方。

在这个结构中，Qwen3ForCausalLM 负责计算 hidden states，Sampler 负责根据 logits 选择下一个 token。Engine 把两者连接起来，形成完整的生成循环。



## 2. 初始化

`Engine.__init__` 分四步完成初始化：

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

部分模型的 tokenizer 没有定义 `pad_token`。如果后续需要构造批量输入，缺少 `pad_token` 会导致填充逻辑报错。这里将其设置为 `eos_token`，是一种常见兼容处理。

`eos_token_id` 会保存下来，在生成循环中用于判断是否停止。

### 2.2 设备与精度

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

v0 固定使用 `cuda:0`，不包含多设备逻辑。

计算精度按以下顺序确定：

1. 如果用户显式指定了 `dtype`，使用用户配置；
2. 否则使用模型配置中的 `torch_dtype`；
3. 如果配置不可用，回退到 `torch.float16`。

### 2.3 模型加载

```py
self.model = Qwen3ForCausalLM(config.hf_config)
load_model(self.model, config.model)
self.model = self.model.to(dtype=self.dtype, device=self.device)
```

模型先在 CPU 上构建，然后加载权重，最后统一移动到目标设备并转换精度。

这个顺序可以避免在权重加载过程中反复在 CPU 和 GPU 之间搬运张量。权重写入完成后，再执行一次 `.to()`，整体更简单，也更容易控制显存。

权重加载逻辑见 [05-weights.md](./05-weights.md)。


### 2.4 Sampler

```py
self.sampler = Sampler().to(device=self.device)
```

`Sampler` 继承自 `nn.Module`，但没有可学习参数。这里仍然调用 `.to(device)`，是为了保证后续在 `generate` 中接收 GPU 张量时，设备状态一致。



## 3. 生成阶段 `generate`

`generate` 负责整个生成循环：编码输入，调用模型，计算 logits，采样下一个 token，并将结果逐步返回。

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

`@torch.inference_mode()` 会关闭梯度跟踪，并跳过 autograd 相关机制。推理阶段不需要反向传播，因此可以减少额外开销。

### 3.1 输入处理

`prompt` 支持两种输入形式：

- 字符串；
- 已经编码好的 `list[int]`。

如果输入是字符串，Engine 会调用 `tokenizer.encode` 得到 token id。如果输入已经是 token id 列表，则直接作为 `input_ids` 使用，便于测试时绕过 tokenizer。

如果输入长度超过模型上下文长度，v0 会从左侧截断，只保留最后 `context_len` 个 token。

### 3.2 采样参数

```py
def _t(v, dtype):
    return None if v is None else torch.tensor([v], device=self.device, dtype=dtype)

temp_t = None if temperature <= 0 else _t(temperature, torch.float)
```

采样参数在进入循环前转换成 GPU 张量，避免每步生成时重复创建。

当 `temperature <= 0` 时，`temp_t` 设为 `None`。Sampler 检测到所有采样参数都为 `None` 后，会直接使用 `argmax` 选择概率最高的 token。


### 3.3 流式输出与停止条件

```py
yield self.tokenizer.decode([next_id])
```

`generate` 是一个生成器函数。每生成一个 token，就通过 `yield` 返回一次文本。调用方可以逐 token 接收，而不需要等待整段生成结束。

停止条件有三个：

1. 采样到 `eos_token_id`；
2. 当前序列长度达到 `context_len`；
3. 生成轮数达到 `max_new_tokens`。

前两个条件会在循环内部判断，满足后提前退出。第三个条件由 `for` 循环本身控制。

由于每次只解码一个 token，而某些 token 可能只是完整字符的一部分，因此流式输出时可能短暂出现字符显示不完整的情况。这是 tokenizer 分词方式造成的，不是 Engine 的问题。


## 5. v0 的范围

v0 的目标是实现一条基本的推理链路：输入文本，模型前向，采样，流式输出。

以下能力不在 v0 中实现：

| 能力 | v0 当前处理 | 后续方向 |
|---|---|---|
| KV Cache | 每步重算完整序列 | 缓存历史 K/V，只计算新增 token |
| 批处理 | 固定 batch size 为 1 | 支持多请求批处理 |
| 显存管理 | 无显存块管理 | PagedAttention |
| 请求调度 | 同步单请求 | 请求队列与连续批处理 |
| 分布式 | 单卡 `cuda:0` | 张量并行、流水线并行 |

v0 先保证流程正确。后续优化主要集中在两个地方：模型前向内部，以及 Engine 的调度层。

## 6. 一次完整调用的路径

以 `run.py` 为例：

```text
EngineConfig(model="~/huggingface/Qwen3-0.6B")
    │
    ▼ Engine.__init__
    │   tokenizer ← AutoTokenizer
    │   model     ← Qwen3ForCausalLM + 权重
    │   sampler   ← Sampler
    │
    ▼ engine.generate("Hello, my name is", max_new_tokens=400)
    │   encode → input_ids
    │   loop:
    │     model.forward
    │     compute_logits
    │     sampler
    │     decode / yield
    │     append next_id
    │   stop:
    │     eos / context_len / max_new_tokens
    │
    ▼ run.py 输出结果并统计速度
```

这一章完成后，Engine 已经可以完成从文本输入到流式输出的基本流程。
下一章处理 KV Cache，解决每步重复计算完整序列的问题。