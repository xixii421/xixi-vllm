# Qwen架构

> [Qwen3整体介绍](https://github.com/datawhalechina/tiny-universe/tree/main/content/Qwen-blog) | [Qwen3模型解析](https://github.com/datawhalechina/self-llm/blob/master/models/Qwen3/01-Qwen3-%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84%E8%A7%A3%E6%9E%90-Blog.md) | 

> [](../src/model.py)

![Qwen3模型架构](./posts/qwen3.png)

## 1. Qwen3-32B中的B代表的是什么？
无论是0.6B、1.5B、7B，它们都共享GQA、SwiGLU、RoPE、Attention、RMSNorm等核心机制，因此从代码的角度来说，它们基本是完全一致的。

但是在`Qwen3Config`的超参数配置上，它们有一些区别：

| 维度 | 0.6B (端侧) | 7B/14B (边缘) | 72B (云端) | 架构影响分析 |
| :--- | :--- | :--- | :--- | :--- |
| **层数** | 约 24 层 | 约 32 - 40 层 | 80 层以上 | 层数决定了模型的逻辑推理深度和“世界观”的复杂度。 |
| **隐藏维度** | 1024 或 1536 | 4096 / 5120 | 8192 以上 | 维度决定了模型**知识容量的宽度**。 |
| **注意力头数** | 16 或 24 | 32 / 40 | 64 以上 | 头数越多，模型能同时关注的**特征子空间**就越丰富。 |
| **GQA 分组比例** | 可能更激进 (如 8个Q共享1个KV) | 标准 (如 4个Q共享1个KV) | 接近 MHA | **小模型为了省显存，会进一步压榨 KV Cache**，牺牲极微小的精度换取端侧的流畅度。 |
| **Head Dim (单头维度)** | 可能非标准 (如 64 或 96) | 标准 (128) | 标准 (128) | 代码中 `head_dim=getattr(config, "head_dim", None)` 就是为了兼容小模型的特殊设计。 |


很多同学会问：“既然 72B 最聪明，为什么还要费力去训 0.6B 和 1.5B 的小模型？” 
实际上是多种因素叠加的结果。

1. **云边端协同**。0.6B的小模型量化后通常占用不到1G内存，可以在手机上快速部署。7B则在单张显卡上，或者本地mac上就可以成功部署，这是本地开发者比较友好的尺寸；而72B+甚至更大的模型，则通常部署在云端，提供更加复杂的推理逻辑。
2. **风洞**。训练一个 72B 的模型需要数千万美元的算力，且周期长达数月。如果直接在大模型上测试新的架构、例如引入一些新的特性，试错成本极高。而小模型则可以快速验证 Loss 曲线更优、收敛更快的技术，才会被最终迁移到更大的旗舰模型上。
3. **全家桶战略**。实际上，现在开源模型中，Qwen几乎可以说是一家独大，生态位明显比单个模型智力更加重要。如果开源不同大小的模型，则开发者只需要学一种架构，就能够在云边端高效切换，并且72B生成出来的数据去蒸馏小模型，也会在特定的领域更加高效。


## 2. 把Layer中的内容组合成Qwen3
```
❯ python test/test_qwen.py
1. 正在读取模型配置 (Config)...
隐藏层维度      : 1024
注意力头数      : 16
KV头数          : 8
层数            : 28
词表大小        : 151936
--------------------------------------------------
2. 正在构建模型骨架...
3. 打印模型架构树:
Qwen3ForCausalLM(
  (model): Qwen3Model(
    (embed_tokens): Embedding(151936, 1024)
    (layers): ModuleList(
      (0-27): 28 x Qwen3DecoderLayer(
        (self_attn): Qwen3Attention(
          (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
          (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
          (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
          (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
        )
        (mlp): Qwen3MLP(
          (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (up_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (down_proj): Linear(in_features=3072, out_features=1024, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
        (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
      )
    )
    (norm): Qwen3RMSNorm((1024,), eps=1e-06)
    (rotary_emb): Qwen3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=1024, out_features=151936, bias=False)
)
```

通过`test/test_qwen`就可以打印查看现在Qwen3-0.6B的整个架构是怎么样的。

## 3. 模型权重加载
在编写模型之前，我们先要做第一件事，现在厂商已经帮我们训练好各种模型了，那我们如何把模型加载到自己写的架构上呢？

> [model_loader代码](../src/model_loader.py)

```py

def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def safetensors_weights_iterator(hf_folder: str) -> Iterable[tuple[str, torch.Tensor]]:
    files = glob.glob(os.path.join(hf_folder, "*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {hf_folder}")
    for st_file in tqdm(files, desc="Loading weights"):
        with safe_open(st_file, framework="pt") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def load_model(model: nn.Module, hf_folder: str) -> nn.Module:
    model.load_weights(safetensors_weights_iterator(hf_folder))
    return model.eval()

```

先看第一个函数`default_weight_loader`，注意这里用的是`param.data.copy_()`，在 PyTorch 中，如果直接使用 `=` 赋值会改变对象的引用，有可能破坏原有的计算图，或者导致 `Parameter` 对象丢失 `requires_grad` 等属性。通过 `.data.copy_()` 原地拷贝，就可以直接**将底层的 Tensor 内存数据进行覆盖**。

再来看流式加载`safetensors_weights_iterator`。

> [!NOTE] 什么是 Safetensors 格式？
传统的 `.bin(Pickle)` 格式在反序列化时存在安全漏洞（对我们来说可能没区别？），并且加载速度慢（这个是真的！）。而 `safetensors` 是一种纯数据格式，不仅安全，并且基于内存映射，加载速度很快。

> [!NOTE] 什么是惰性加载与 mmap？
`safe_open` 并不会一次性把整个文件读入内存，它利用操作系统的 `mmap` 技术，将硬盘的文件直接映射到虚拟内存，只有当调用 `f.get_tensor(name)` 时，才会触发磁盘 IO 读取这一部分数据。

> [!NOTE] 这里为什么要用 yield？
这里把函数写成了**生成器（Generator）**。`yield` 和 `return` 最大的区别在于：`return` 会一次性把所有结果打包返回；而 `yield` 每执行一次，就"吐"出一个 `(name, tensor)` 对，然后**暂停**在这里，等调用方消费完再继续。

这样做是防止**把几百 GB 的权重一次性塞进内存。**

以 Qwen3-0.6B 为例，权重大约 1.2 GB；但如果是 72B，光权重文件就有 140 GB 以上，而且通常被切成多个 `.safetensors` 分片。如果用 `return` 把所有 tensor 收集到一个 `list` 里再返回，内存峰值 ≈ 全部权重之和。还没开始推理，内存就先爆了。

用 `yield` 之后，整个加载链路变成了**流水线**：

```
磁盘 (.safetensors)
  │  mmap 映射，不立刻读入
  ▼
safe_open → f.get_tensor(name)    ← 按需读取单个 tensor
  │  yield (name, tensor)          ← 吐出一个，暂停
  ▼
load_weights 的 for 循环           ← 立刻 copy_ 到模型参数
  │  default_weight_loader / QKV 合并
  ▼
模型 param.data 就位               ← 这个 tensor 可以被 GC 回收
```

这意味着在任意时刻，内存里**最多只躺着当前正在处理的那一个权重 tensor**，处理完就释放，峰值内存从 O(全部权重) 降到了 O(单个最大 tensor)。

还有一个工程上的好处是**解耦**。`safetensors_weights_iterator` 只负责"从磁盘一个一个吐权重"，`load_weights` 只负责"接到一个、写入一个"。以后如果要支持从网络流式下载、或者从量化后的不同格式加载，只需要换一个新的 iterator，消费端的 `load_weights` 完全不用改。这正是 vLLM 真实代码里的设计思路。

### 3.1 把权重填进模型骨架：`load_weights`

iterator 吐出来的 `(name, tensor)` 对，最终要落到我们搭好的模型骨架上。`Qwen3ForCausalLM.load_weights` 就是干这件事的：

```py
def load_weights(self, weights):
    params = dict(self.named_parameters())
    loaded = set()

    for name, w in weights:                          # ← 消费 yield 出来的每一个权重
        # HuggingFace 存了 q_proj / k_proj / v_proj 三个矩阵
        # 我们推理时合并成了一个 qkv_proj，需要按偏移量拼回去
        for hf, our, offset, size in [
            ("q_proj", "qkv_proj", 0, _q_size),
            ("k_proj", "qkv_proj", _q_size, _kv_size),
            ("v_proj", "qkv_proj", _q_size + _kv_size, _kv_size),
        ]:
            if hf in name:
                p = params[name.replace(hf, our)]
                p.data[offset:offset + size].copy_(w)  # 切片写入，不是整段覆盖
                break
        else:
            if name in params:
                default_weight_loader(params[name], w)  # 名字对得上，直接拷贝
```

这里有两个值得注意的细节：

1. **QKV 合并加载**。HuggingFace 的 checkpoint 里，`q_proj`、`k_proj`、`v_proj` 是三个独立的权重文件条目；但我们在 `Qwen3Attention` 里为了推理效率，把它们合并成了一个 `qkv_proj`。所以加载时不能整段 `copy_`，而是按偏移量把 Q、K、V 分别塞进 `qkv_proj` 的前、中、后三段。

2. **名字映射与跳过**。HF 的权重名和我们 `named_parameters()` 的名字不一定完全一致（比如 `model.layers.0.self_attn.q_proj.weight` vs 我们内部的命名）。对不上的权重会被记录到 `skipped` 列表里打印出来，方便排查。

把 `safetensors_weights_iterator` 和 `load_weights` 串起来，完整的加载流程就是：

```py
# model_loader.py
load_model(model, hf_folder)
  └─ model.load_weights(safetensors_weights_iterator(hf_folder))
       └─ for (name, tensor) in iterator:   # yield 逐个吐出
            └─ copy_ 到对应 param             # 消费一个、释放一个
```


## 4. Qwen3 模型实现

> [model.py 源码](../src/model.py)

<span style="color:#1d4ed8">**模块名**</span> · <span style="color:#b45309">**核心机制**</span> · <span style="color:#059669">**实现细节**</span> · <span style="color:#7c3aed">**张量形状**</span>

`model.py` 把前面各层积木拼成完整 Qwen3。

### 4.1 四层嵌套骨架

整个文件只有四个 `nn.Module`，自底向上嵌套

```
Qwen3ForCausalLM          ← 对外入口，挂 lm_head 与 load_weights
  └─ Qwen3Model           ← 词嵌入 + N 层 Decoder + 最终 Norm
       └─ Qwen3DecoderLayer × N
            ├─ Qwen3Attention
            └─ Qwen3MLP
```

<span style="color:#059669">**细节**</span> `Qwen3ForCausalLM.forward` 只调 `self.model`，不碰 `lm_head`。推理引擎在拿到 hidden states 后再调 `compute_logits`，把**表征计算**和**词表投影**拆开，后续做 KV Cache 或张量并行时只需改 backbone，lm_head 可独立处理。

### 4.2 Qwen3MLP

```py
class Qwen3MLP(nn.Module):
    def forward(self, x):
        return self.down_proj(
            self.act_fn(torch.cat([self.up_proj(x), self.gate_proj(x)], dim=-1))
        )
```

<span style="color:#b45309">**机制**</span> 标准 Transformer FFN 的 SwiGLU 变体。`gate_proj` 与 `up_proj` 各做一次线性映射，在最后一维拼接后送入 `SiluAndMul`，等价于 `SiLU(gate) × up`，再经 `down_proj` 压回隐藏维。

<span style="color:#059669">**细节**</span> HF 权重里 `gate_proj` 和 `up_proj` 仍是两个独立矩阵，forward 时才 `cat`。这和 Attention 里 QKV 的**权重级合并**不同，这里只是计算语义上的拼接，加载逻辑无需特殊处理。

<span style="color:#7c3aed">**形状**</span> 以 0.6B 为例，<span style="color:#7c3aed">`x`</span> 为 `[batch, seq, 1024]`，中间维膨胀到 3072，再收回 1024。

### 4.3 Qwen3Attention（GQA + QK Norm + RoPE）

这是 `model.py` 里最重的一块，分**初始化**和**前向**两段看。

**初始化阶段**

`Qwen3Attention` 直接吃 `Qwen3Config`，不再逐参数传递：

```py
class Qwen3Attention(nn.Module):
    def __init__(self, config):
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = nn.Linear(
            config.hidden_size, self.q_size + 2 * self.kv_size, bias=False
        )
        self.q_norm = RMSNorm(self.head_dim, ...)
        self.k_norm = RMSNorm(self.head_dim, ...)
```

<span style="color:#b45309">**机制**</span> 三个投影矩阵合成一个 `qkv_proj`，一次 GEMM 同时产出 Q、K、V。`q_norm` / `k_norm` 是 Qwen3 特有的 **QK Norm**，对每个 head 内的向量单独做 RMSNorm，训练更稳。

<span style="color:#059669">**细节**</span> `head_dim` 允许从 config 显式传入。小模型可能是 64 或 96，不总等于 `hidden_size // num_heads`。`getattr(config, "head_dim", None)` 就是为兼容非标准 head 维。

**前向阶段**

```py
qkv = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

q = self.q_norm(q.unflatten(-1, (-1, self.head_dim))).flatten(-2)
k = self.k_norm(k.unflatten(-1, (-1, self.head_dim))).flatten(-2)

q, k = self.rotary_emb(positions, q, k)

q = q.view(..., self.num_heads, self.head_dim)
k = k.view(..., self.num_kv_heads, self.head_dim)
v = v.view(..., self.num_kv_heads, self.head_dim)

out = self.attn(q, k, v)
return self.o_proj(out.flatten(-2))
```

数据流可以概括为五条

1. <span style="color:#1d4ed8">`qkv_proj`</span> 一次线性，再 `split` 切成 Q / K / V 三段
2. <span style="color:#1d4ed8">`unflatten → norm → flatten`</span> 把最后一维拆成 `[num_heads, head_dim]` 做 QK Norm，再拍平回去
3. <span style="color:#1d4ed8">`rotary_emb`</span> 只旋转 Q 和 K，V 不动
4. <span style="color:#1d4ed8">`view`</span> 显式分出 head 维，Q 有 16 头，K/V 只有 8 头（GQA）
5. <span style="color:#1d4ed8">`Attention`</span> 内部完成 GQA 广播与 SDPA，输出再经 `o_proj`

<span style="color:#059669">**细节**</span> `positions` 从外层一路传进来，不在 Attention 内部生成。v0 引擎每次把完整序列的位置 `[0, 1, …, seq_len-1]` 传入，为后续只传增量位置、接入 KV Cache 留好接口。

<span style="color:#059669">**细节**</span> `scaling = head_dim ** -0.5` 在初始化时就算好，避免 forward 重复开方。

### 4.4 Qwen3DecoderLayer

```py
def forward(self, positions, hidden_states, residual):
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(positions, hidden_states)
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    return self.mlp(hidden_states), residual
```

<span style="color:#b45309">**机制**</span> Pre Norm 结构。每层先 Norm 再 Attention / MLP，残差在 Norm 内部融合。

<span style="color:#059669">**细节**</span> 第一层 `residual is None` 时，把 embedding 输出存为 residual，只对 hidden_states 做 Norm。从第二层起，`input_layernorm(hidden_states, residual)` 在**一个算子里**完成 `h + residual` 再归一化，少一次中间张量分配。`RMSNorm` 的 fused 接口在 `layers/layernorm.py` 里实现，这里只负责调度。

<span style="color:#059669">**细节**</span> MLP 子层没有第三个 Norm。Attention 后的 `post_attention_layernorm` 已经融合了残差，MLP 直接吃归一化后的结果。

子层调用顺序

```
input_layernorm → self_attn → post_attention_layernorm → mlp
```

### 4.5 Qwen3Model

```py
def forward(self, input_ids, positions):
    h, residual = self.embed_tokens(input_ids), None
    for layer in self.layers:
        h, residual = layer(positions, h, residual)
    return self.norm(h, residual)[0]
```

<span style="color:#b45309">**机制**</span> 标准 Decoder Only 堆叠。`embed_tokens` 把 token id 映射到隐藏维，28 层 `Qwen3DecoderLayer` 逐层传递 `(h, residual)` 元组，最后 `norm` 输出最后一层表征。

<span style="color:#059669">**细节**</span> `ModuleList` 在 `__init__` 里用列表推导一次性建好全部层，`layer_id` 参数目前未使用，预留给将来的层间差异化（如 MoE 路由、层间 KV 共享）。

<span style="color:#059669">**细节**</span> 最终 `self.norm(h, residual)[0]` 只取元组第一个元素。最后一层 MLP 输出还没和 residual 合并，由 final norm 完成这步融合。

### 4.6 Qwen3ForCausalLM

```py
class Qwen3ForCausalLM(nn.Module):
    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
```

<span style="color:#b45309">**机制**</span> Causal LM 的标准两头结构。`model` 产出 hidden states，`lm_head` 线性投影到词表大小，得到每个位置的 logits。

<span style="color:#059669">**细节**</span> `lm_head` 与 `embed_tokens` **不共享权重**。部分模型会做 tying，Qwen3 默认分开存储，加载时各写各的。

<span style="color:#059669">**细节**</span> `load_weights` 已在第 3 节讲过。它与 forward 路径解耦，消费 `yield` 出来的权重流，按名匹配或 QKV 切片合并，是模型能跑起来的最后一步。

### 4.7 一个 token 穿过 model.py 的路径

```
input_ids [1, seq]
    │
    ▼ embed_tokens          →  h [1, seq, 1024]
    │
    ▼ DecoderLayer × 28
    │   input_norm → Attention(qkv→QKnorm→RoPE→GQA→o_proj)
    │   post_norm  → MLP(gate∥up→SiLU×→down)
    │
    ▼ final RMSNorm          →  h [1, seq, 1024]
    │
    ▼ lm_head (引擎侧调用)    →  logits [1, seq, 151936]
```

至此，`model.py` 从空骨架到可加载权重、可 forward 的完整 Qwen3 闭环。