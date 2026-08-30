# 模型权重加载

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
> 传统的 `.bin(Pickle)` 格式在反序列化时存在安全漏洞（对我们来说可能没区别？），并且加载速度慢（这个是真的！）。而 `safetensors` 是一种纯数据格式，不仅安全，并且基于内存映射，加载速度很快。

> [!NOTE] 什么是惰性加载与 mmap？
> `safe_open` 并不会一次性把整个文件读入内存，它利用操作系统的 `mmap` 技术，将硬盘的文件直接映射到虚拟内存，只有当调用 `f.get_tensor(name)` 时，才会触发磁盘 IO 读取这一部分数据。

> [!NOTE] 这里为什么要用 yield？
> 这里把函数写成了**生成器（Generator）**。`yield` 和 `return` 最大的区别在于：`return` 会一次性把所有结果打包返回；而 `yield` 每执行一次，就"吐"出一个 `(name, tensor)` 对，然后**暂停**在这里，等调用方消费完再继续。

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