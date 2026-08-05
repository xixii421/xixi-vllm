# 07 · v3 Attention 相对 v2 的性能优化

> 对应 [01-v2性能瓶颈](./01-v2性能瓶颈.md) 中 **瓶颈 A（Attention）** 和 **瓶颈 C（KV 写）**。  
> v3 用 **FlashInfer + Triton** 两把刀，把 v2 的 PyTorch SDPA 体系彻底换血。

v3 对 attention 做了系统性的替换，不仅是算子替换，连 KV 布局、数据流、上下文传递方式都改了。


> 这里为什么我没有自己手动实现呢？——自己手动实现性能太差，后续考虑做flash-attn的v1-vx的教学版


## 一、对照总表


| #   | 优化点                | v2                             | v3                                                | 收益类型                                 |
| --- | ------------------ | ------------------------------ | ------------------------------------------------- | ------------------------------------ |
| 1   | Attention 后端       | PyTorch `SDPA`                 | FlashInfer paged (C++/CUDA)                       | **计算量 + 显存带宽**                       |
| 2   | Decode 合批          | `for` 逐样本 SDPA → `cat`         | `BatchDecodeWrapper.run(q, paged_kv)` 整批一次        | **消除 Python 循环 + N 次 kernel launch** |
| 3   | Prefill with cache | `gather` 全量 KV → 拼接 → 材料化 mask | FlashInfer varlen paged prefill，直接读 paged KV      | **消除 gather + mask**                 |
| 4   | KV 写入              | `index_copy_`（scatter-like）    | Triton `store_kvcache`（fused kernel）              | **消除 scatter 开销**                    |
| 5   | KV 布局              | `[L, N_blocks, B, Hkv, D]`     | `[2, L, blocks, block_size, n_kv, head_dim]`（NHD） | **对齐 FlashInfer**                    |
| 6   | GQA 展开             | 显式 `_expand_gqa()` 材料化         | FlashInfer 内核原生处理                                 | **消除 broadcasting 开销**               |
| 7   | 上下文传递              | 模块级 `global` 变量                | `contextvars.ContextVar` + context manager        | **线程安全 + CG 兼容**                     |
| 8   | 性能旋钮               | 无                              | `use_tensor_cores` / `non_blocking`               | **可选加速**                             |


---




## 二、逐个优化详解




### 优化 1：Attention 后端从 SDPA 换成 FlashInfer（最核心）

这是 v2 → v3 最大的单个性能改动。

**v2 实现：**

```python
# v2 — layers/attention.py _sdpa()
def _sdpa(self, q, k, v, *, is_causal, attn_mask=None):
    kwargs = {"scale": self.scaling}
    if attn_mask is not None:
        kwargs["attn_mask"] = attn_mask
    else:
        kwargs["is_causal"] = is_causal
    if _SDPA_HAS_GQA and k.shape[1] != q.shape[1]:
        kwargs["enable_gqa"] = True
        return F.scaled_dot_product_attention(q, k, v, **kwargs)
    if k.shape[1] != q.shape[1]:
        k, v = self._expand_gqa(k, v)
    return F.scaled_dot_product_attention(q, k, v, **kwargs)
```

`F.scaled_dot_product_attention` 是 PyTorch 的统一入口，底层由 cuDNN / FlashAttention / Memory-Efficient Attention 等多个后端中择优。问题在于：**它不认 Paged KV Cache，要求传入的 K/V 是连续张量**。v2 只能先把 paged KV gather 成连续张量再传入，这就引出了优化 2 和优化 3。

**v3 实现：**

```python
# v3 — layers/attention.py forward()
def forward(self, q, k, v):
    meta = _get_metadata()
    pool = meta.kv_cache_pool
    k_4d = pool[0, self.layer_id]  # paged KV，不 gather
    v_4d = pool[1, self.layer_id]

    store_kvcache(k, v, k_4d.reshape(...), v_4d.reshape(...), meta.slot_mapping)

    q_3d = q.view(-1, self.num_heads, head_dim)
    o = meta.wrapper.run(q_3d, (k_4d, v_4d))  # ← FlashInfer 直接读 paged KV
    return o.view(-1, self.num_heads * head_dim)
```

`meta.wrapper.run(q, (k_4d, v_4d))` 背后是 FlashInfer 的 C++/CUDA 内核，它**原地读取 paged KV**，不需要 gather。wrapper 在 `build_metadata()` 中已经用 `plan()` 绑定好了 paged KV 索引结构（indptr/indices/last_page_len）。

```python
# v3 — attention_backend.py build_metadata()
# prefill 分支：
self.prefill_wrapper.plan(
    qo_indptr=cu_seqlen_q_cpu,
    paged_kv_indices=paged_kv_indices_gpu,
    paged_kv_indptr=paged_kv_indptr_cpu,
    paged_kv_last_page_len=paged_kv_last_page_len_cpu,
    ...
)
# decode 分支：
self.decode_wrapper.plan(
    indptr=paged_kv_indptr_cpu,
    indices=paged_kv_indices_gpu,
    last_page_len=paged_kv_last_page_len_cpu,
    ...
)
```

**效果：** FlashInfer 的 paged attention 内核以 NHD 布局直接索引物理 KV 页面，省掉了一次全量 KV gather 的内存分配和拷贝，也避免 SDPA 内部的 QKV 重排。这是 **瓶颈 A 的核心解法**。

---



### 优化 2：Decode 从逐样本 SDPA 循环到批处理 paged kernel

这是 v2 decode 阶段最严重的性能反模式。

**v2 实现（**`_decode`**）：**

```python
def _decode(self, q, ctx):
    outs = []
    for b in range(q.shape[0]):          # ← 逐样本循环！
        ctx_len = ctx.context_lens[b]
        k_c, v_c = ctx.store.gather(     # ← 每样本 gather KV
            self.layer_id, ctx.block_tables[b], ctx_len
        )
        k_c = k_c.unsqueeze(0).transpose(1, 2)
        v_c = v_c.unsqueeze(0).transpose(1, 2)
        outs.append(self._sdpa(q[b:b+1], k_c, v_c, is_causal=False))
    return torch.cat(outs, dim=0).transpose(1, 2).contiguous()
```

每一层、每一个 decode token、每一次 forward，都要：

1. **gather** 该序列的全量 KV（从 paged 池里索引出连续张量）
2. **transpose** 成 SDPA 期望的 `[B, H, S, D]` 格式
3. **单独调用一次 SDPA**
4. **cat** 拼接回来

batch_size=8 的 decode 要跑 8 次 SDPA + 8 次 gather + 1 次 cat，**kernel launch 数量随 batch 线性增长**。

**v3 实现：**

```python
# v3 — 直接一个 run 调用
o = meta.wrapper.run(q_3d, (k_4d, v_4d))
```

`BatchDecodeWithPagedKVCacheWrapper.run()` 一次处理整个 batch 的所有 decode 请求。FlashInfer 内核内部并行处理所有序列的 paged attention，**O(1) 次 kernel launch，不随 batch 增长**。

**量化收益：** batch=8 时 v2 要 per layer per forward 至少 24 次 kernel launch（8 gather + 8 SDPA + 8 transpose/cat），v3 只要 **1 次**。如果 batch=256，v2 的 launch 数量是 v3 的 768 倍。即使单个 SDPA 很快，launch overhead 在小模型（Qwen3-0.6B）上会占据 decode 延迟的主要部分。

---



### 优化 3：Prefill with cache 消除 gather + 显式 mask 

v2 的 prefill 中，当序列已有缓存的 KV 前缀时，需要做"增量 prefill"：把旧的 KV 和新的 KV 拼起来，然后构建一个不规则因果 mask。

**v2 实现（**`_prefill`**）：**

```python
def _prefill(self, q, k, v, ctx):
    cached = ctx.cached_lens[0]
    if cached == 0:
        return self._sdpa(q, k, v, is_causal=True)

    # gather 全量历史 KV
    k_full, v_full = ctx.store.gather(
        self.layer_id, ctx.block_tables[0], ctx_len
    )
    k_full = k_full.unsqueeze(0).transpose(1, 2)  # [1, H, ctx_len, D]
    v_full = v_full.unsqueeze(0).transpose(1, 2)

    # 材料化一个 [q_len, ctx_len] 的 attention mask！
    qi = torch.arange(q_len, device=device).unsqueeze(1) + cached
    kj = torch.arange(ctx_len, device=device).unsqueeze(0)
    allow = qi >= kj
    attn_mask = torch.zeros(q_len, ctx_len, device=device, dtype=q.dtype)
    attn_mask = attn_mask.masked_fill(~allow, -inf)

    return self._sdpa(q, k_full, v_full, is_causal=False, attn_mask=attn_mask)
```

两个大问题：

1. **gather**：把整条序列的历史 KV 从 paged 池里读出来，分配一个 `[ctx_len, H, D]` 大小的连续张量，引发显存分配和拷贝
2. **材料化 mask**：创建一个 `[q_len, ctx_len]` 的 float 张量。如果 ctx_len=8192，q_len=4096，这就是一个 32MB 的 mask 张量，在整批的所有层都要分配一次

**v3 实现：**

```python
# v3 — layers/attention_backend.py build_metadata() prefill 分支
self.prefill_wrapper.plan(
    qo_indptr=cu_seqlen_q_cpu,        # ← varlen 的 Q 索引
    paged_kv_indices=paged_kv_indices_gpu,
    paged_kv_indptr=paged_kv_indptr_cpu,
    paged_kv_last_page_len=paged_kv_last_page_len_cpu,
    causal=True,                        # ← 内核内部处理因果 mask
    ...
)
```

FlashInfer 的 `BatchPrefillWithPagedKVCacheWrapper` 支持 varlen prefill + paged KV：

- `cu_seqlen_q` 标记每个序列的 Q 长度，支持不同序列不同 query 长度
- `causal=True` 时，内核内部根据 Q/K 的全局位置关系决定可见性，**不需要显式 mask 张量**
- paged KV 直接原位读取，不需要 gather

**效果：** 消除 gather 的显存分配/拷贝 + 消除 mask 张量的材料化。在长序列 prefill 场景下，v2 的 gather + mask 开销占比可达 30%+，v3 直接消灭这两项。

---



### 优化 4：KV 写入从 index_copy_ 到 Triton fused kernel

**v2 实现（**`PagedStore.write`**）：**

```python
def write(self, layer_id, slot_mapping, k, v):
    k, v = _coerce_kv_token_major(k, v)  # flatten [B,S,H,D] → [B*S, H*D]
    flat_k, flat_v, flat_dim = _flat_layer_views(self.pool, layer_id)
    flat_k.index_copy_(0, slot_mapping, k.reshape(n, flat_dim).detach())
    flat_v.index_copy_(0, slot_mapping, v.reshape(n, flat_dim).detach())
```

`index_copy_` 是 PyTorch 的 scatter-like 操作：对于每个 `slot_mapping[i]`，把 `k[i]` 复制到 `flat_k[slot_mapping[i]]`。这是一种间接写操作，需要先 flatten 再 scatter，内存访问模式不连续。

**v3 实现（Triton** `store_kvcache`**）：**

```python
@triton.jit
def _store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride,
                           k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
                           D, BLOCK_SIZE):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return         # ← -1 表示 padding，跳过
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    key = tl.load(key_ptr + idx * key_stride + offsets, mask=mask)
    value = tl.load(value_ptr + idx * value_stride + offsets, mask=mask)
    tl.store(k_cache_ptr + slot * D + offsets, key, mask=mask)
    tl.store(v_cache_ptr + slot * D + offsets, value, mask=mask)
```

Triton 内核的优化点：

1. **一个 token 一个 program**：grid 大小 = n（token 数），每个 CUDA thread block 处理一个 token 的 key+value 写入
2. **向量化 load/store**：`BLOCK_SIZE` 是 head_dim×num_heads 的 2 的幂，一个 token 的 K 和 V 各一次向量 load + 一次向量 store
3. **padding 跳过**：`slot == -1` 时直接 return，不写入（v2 的 index_copy_ 也会跳过，但需要 index_copy_ 内部处理）
4. **无 flatten 开销**：key/value 按 stride 直接读取，不需要 reshape
5. **key 和 value 在同一内核中写入**：减少一次 kernel launch

---



### 优化 5：KV 布局从 `[L, N_blocks, B, Hkv, D]` 到 `[2, L, blocks, block_size, n_kv, head_dim]`

**v2 布局：**

```python
# 5D: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
self.k = torch.zeros(shape, device=device, dtype=dtype)
self.v = torch.zeros(shape, device=device, dtype=dtype)
```

`gather()` 用高级索引 `pool.k[layer_id, physical, offsets]` 读取。

**v3 布局：**

```python
# 6D: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
# 第 0 维 0=k, 1=v
```

NHD（num_heads × head_dim interleaved）布局是 FlashInfer 的原生格式。`wrapper.run(q, (k_4d, v_4d))` 中的 K/V 就是 4D 的 `[num_blocks, block_size, num_kv_heads, head_dim]` 视图。

**收益：** 对齐 FlashInfer 的内部布局，避免运行时 layout 转换。

---



### 优化 6：GQA 展开消除

**v2：**

```python
def _expand_gqa(self, k, v):
    if self.num_kv_groups <= 1:
        return k, v
    b, hkv, s, d = k.shape
    k = k.unsqueeze(2).expand(b, hkv, self.num_kv_groups, s, d).reshape(
        b, self.num_heads, s, d)
    v = v.unsqueeze(2).expand(b, hkv, self.num_kv_groups, s, d).reshape(
        b, self.num_heads, s, d)
    return k, v
```

对于 GQA（如 Qwen3 的 16 Q heads : 8 KV heads），`_expand_gqa` 把 K/V 从 `[B, 8, S, D]` 广播成 `[B, 16, S, D]`，**材料化一份 2 倍大的 K/V**。在 prefill 长序列时，这个广播带来的显存分配和带宽开销不可忽略。

**v3：** FlashInfer 内核原生支持 GQA，`plan()` 时传入 `num_qo_heads` 和 `num_kv_heads`，内核内部处理 head 映射。

---



### 优化 7：上下文传递从 global 变量到 contextvars

**v2（**`kv/context.py`**）：**

```python
_forward_ctx: ForwardContext | None = None  # 模块级全局

def set_forward_context(ctx):
    global _forward_ctx
    _forward_ctx = ctx
```

**v3（**`layers/attention.py`**）：**

```python
_ATTN_META: contextvars.ContextVar[Optional["AttentionMetadata"]] = (
    contextvars.ContextVar("_ATTN_META", default=None)
)

@contextmanager
def attention_context(meta):
    token = _ATTN_META.set(meta)
    try:
        yield
    finally:
        _ATTN_META.reset(token)
```

`contextvars.ContextVar` 是 Python 3.7+ 的标准库，支持协程和线程安全的上下文隔离。对于 CUDA Graph 回放时可能的多线程场景，`contextvars` 保证每个执行上下文拿到正确的 metadata，不会踩到别的请求的池引用。

v2 的 `set_forward_context` / `get_forward_context` 只适用于单线程 eager 模式。

---



### 优化 8：性能旋钮

v3 新增两个配置项，通过 `BackendConfig` 控制：

```python
@dataclass
class BackendConfig:
    use_tensor_cores: bool = True   # FlashInfer 是否启用 Tensor Core
    non_blocking: bool = True       # plan() 时 H2D 是否异步
```

- `use_tensor_cores=True`：FlashInfer 使用 Tensor Core 加速 attention 中的矩阵乘法，在 A100/H100 上尤其显著
- `non_blocking=True`：`plan()` 调用中的 H2D 拷贝使用 CUDA stream 异步执行，可以与前一步的计算重叠

v2 没有这些控制——SDPA 的后端选择由 PyTorch 内部决定，用户无法直接干预。

---



## 三、数据流对比



### v2 数据流（以 decode 为例）

```text
每层每步：
  q = qkv_proj(hidden)      # [B, H, D]
  k = qkv_proj(hidden)      # [B, Hkv, D]
  v = qkv_proj(hidden)      # [B, Hkv, D]

  # 写 KV
  store.write(layer_id, slot_mapping, k.T, v.T)
    → index_copy_: flat K/V scatter 写入 paged pool

  # 读注意力
  for b in range(B):         # ← 逐样本！
    k_full = store.gather(layer_id, block_table[b], ctx_len)
    out[b] = SDPA(q[b], k_full, v_full)
  out = torch.cat(out)       # ← 拼接
```



### v3 数据流（decode）

```text
每步开始时，build_metadata() 调用一次：
  compute_paged_kv_metadata(seq_lens, block_tables)
    → indptr_cpu, indices_gpu, last_page_cpu
  decode_wrapper.plan(indptr, indices, last_page_len, ...)
    → FlashInfer 绑定 paged KV 索引

每层每步：
  q = qkv_proj(hidden)
  k = qkv_proj(hidden)
  v = qkv_proj(hidden)

  # 写 KV
  store_kvcache(k, v, k_cache_4d, v_cache_4d, slot_mapping)
    → Triton kernel: 每个 token 一个 program，fused K/V write

  # 读注意力
  o = wrapper.run(q, (k_cache_4d, v_cache_4d))
    → FlashInfer: 一次 kernel，整批处理
```

v3 的 decode 从 v2 的 **O(B × L) 次 kernel launch**（每序列每层一次 gather + 一次 SDPA）降低到 **O(L) 次 kernel launch**（每层一次 write + 一次 FlashInfer run）。

## 四、v3 attention 系统的模块架构

```
src/layers/
├── attention.py              # Attention nn.Module + context 管理
│   ├── AttentionMetadata     #    池引用 + wrapper + slot_mapping
│   ├── attention_context()   #    contextvars context manager
│   └── Attention.forward()   #    store_kvcache → wrapper.run
│
├── attention_backend.py      # FlashInfer 后端配置
│   ├── BackendConfig         #    heads / dim / block_size / dtype / TC / async
│   ├── FlashInferBackend     #    管理 prefill/decoder wrapper 生命周期
│   │   └── build_metadata()  #    plan() + 构建 AttentionMetadata
│   └── compute_paged_kv_metadata()
│                              #    从 seq_lens + block_tables 生成 CSR 索引
│
memory/kv_ops.py              # Triton KV 写入内核
└── store_kvcache()            #    每个 token 一个 program，fused K+V write
```

v2 对应的模块分布在 `kv/pool.py`（BlockPool）、`pagedattention/store.py`（PagedStore write/gather）、`pagedattention/slots.py`（slot 映射）、`layers/attention.py`（SDPA Attention），且上下文通过 `kv/context.py` 的全局变量传递。**v3 将这些全部替换为 FlashInfer + Triton 的组合。**





## 六、仍有改进空间

1. **Prefill 长文本无 CG**：v3 的 prefill 首 chunk 走 piecewise CUDA Graph（覆盖 ≤1536 token），更长的续 chunk 走 eager（shape 动态、无法固定图）。曾尝试用 torch.compile 优化 prefill，实测因 FlashInfer `.plan()` 动态 shape 触发 N_layers 次 graph break，反而更慢（44ms vs 22ms），已从 v3 移除。
```
| Prompt |  v4 TTFT  | v4 tok/s  |  v5 TTFT  | v5 tok/s  | v5/v4  |
|--------|-----------|-----------|-----------|-----------|--------|
|   128  |  35.14ms  |   3,610   |  37.81ms  |   3,237   |  0.9x  |
|   256  |  45.31ms  |   5,581   |  39.39ms  |   6,447   |  1.2x  |
|   512  |  67.11ms  |   7,499   |  38.44ms  |  13,206   |  1.8x  |
|  1024  |  98.20ms  |  10,318   |  38.11ms  |  26,874   |  2.6x  |
|  2048  | 134.76ms  |  15,036   |  38.86ms  |  52,686   |  3.5x  |
|  4096  | 218.79ms  |  18,584   |  39.56ms  | 102,860   |  5.5x  |
```
2. **decode prefill 双 forward**：`model_runner.run()` 先 decode 再 prefill，两次独立的 forward，未做 kernel fusion 或 stream 重叠。
3. **KV 池 H2D 同步**：`compute_paged_kv_metadata()` 在 CPU 构建 CSR 索引后 H2D，虽然支持 `non_blocking`，但 prefill 和 decode 的 build_metadata 可以进一步与 model forward 重叠。


期待交给你们去解决。


# 七、FlashAttention vs FlashInfer：为什么 v3 选择了 FlashInfer
> [参考知乎：vllm的多后端](https://zhuanlan.zhihu.com/p/2009763766489866891)

> v3 在 **LM Decode/Prefill 侧使用 FlashInfer**（`attention_backend.py`）；  
> v4 **ViT Attention 侧使用 `flash_attn_varlen_func`**（`vit_attn.py`）。  

## 一、两个库的定位差异

FlashAttention 是一个 **GPU kernel 算法**。它用 tiling + online-softmax 在片上 SRAM 中完成注意力计算，核心创新在 **计算执行顺序**，不涉及内存管理。

FlashInfer是一个 **端到端的注意力引擎**。它底层使用 FlashAttention 风格的 kernel 做实际计算，但在其上增加了一层完整的 **调度、内存管理、编译** 基础设施。


## 二、核心对比
> [这一模块为AI所写，实际上我最明显的感受是，flash-attn安装得在本地编译，经常把CPU核跑满，会把ssh都搞崩，从上手易用性来说，我更喜欢flashInfer]
| 维度 | FlashAttention | FlashInfer |
|:--- |:--- |:--- |
| **设计目标** | 训练 + 长序列 prefill | 推理 serving（decode + paged + continuous batching） |
| **内存布局** | 连续张量 `[B, H, S, D]` | 分页 block-sparse、Ragged、连续皆可 |
| **对 paged KV 的处理** | 早期不支持（需要 gather + cat），FA3 新增 `block_table` 参数 | 原生支持，CSR 索引直接寻址 |
| **Decode 效率** | 通用 kernel，每样本一次 | 合批 kernel，`prefetch page indices` + `Split-K` |
| **CUDA Graph** | 需外部拼接 buffer，兼容性差 | 原生 Wrapper + 静态 buffer 绑定 |
| **编译方式** | 预编译 wheel | JIT 按需编译 |
| **GQA 处理** | 高层 PyTorch 展开或 `enable_gqa` | 内核层原生处理 |
| **MLA/DeepSeek** | 不支持 | 原生 MLA Attention 支持 |
| **安装** | pip install，但需 CUDA 运行时 | pip install，轻量 |
| **工业部署** | vLLM（早期）、HuggingFace transformers | vLLM（默认）、SGLang、TensorRT-LLM、TGI |
| **版本形态** | pip 包 `flash-attn` | pip 包 `flashinfer` |

## 六、参考

- [参考知乎：vllm的多后端](https://zhuanlan.zhihu.com/p/2009763766489866891)
- FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving (MLSys 2025 Best Paper) — [arxiv:2501.01005](https://arxiv.org/abs/2501.01005)
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (NeurIPS 2022) — [arxiv:2205.14135](https://arxiv.org/abs/2205.14135)
- [FlashInfer 官方文档](https://flashinfer.ai/)
- [FlashAttention vs PagedAttention vs FlashInfer 深度分析](https://dreaming.press/posts/flashattention-vs-pagedattention-vs-flashinfer/)
