# Attention：从标准实现到 FlashAttention 与 PagedAttention
为了进一步了解v3究竟是如何对attention进行优化的，我们从最原始的attention开始，分析原本的attention性能瓶颈究竟是什么，然后再看一下FlashAttention是如何把它变快的。

## 1. 标准 attention 的显存瓶颈

一个单头 attention 的数学定义很简洁：

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
```

写成代码如下：

```python
def attention(Q, K, V):                 # Q, K, V: [N, d]
    S = Q @ K.T / (d ** 0.5)            # [N, N] 注意力分数
    P = softmax(S, dim=-1)              # [N, N] 概率
    return P @ V
```

问题藏在这两个中间变量 `S` 和 `P` 里。它们都是 `N × N` 的方阵，N 是序列长度。假设 N=8192、bf16 精度，`S` 和 `P` 各占 128 MB，而且每过一层 transformer 都要完整写回显存再读回来一遍。

attention 的算术量只有 O(N²)，但显存读写量也是 O(N²)。在现在的 GPU 上，算术吞吐的增长速度远快于显存带宽，于是这个 O(N²) 的显存读写就成了真正的瓶颈，计算单元大部分时间在等数据。

### 1.1 roofline 分析

RTX 4090 的关键指标：
- FP32 算力约 82.6 TFLOPS，
- 显存带宽约 1008 GB/s。

两个数字的比值叫做算术强度，单位是FLOP/byte。一个计算密集的算子要做到不浪费算力，每做 1 次 FLOP 至少要搬 `82.6e12 / 1008e9 ≈ 82` 字节以内的数据。

```
I_ridge = 算力 / 带宽 = 82.6e12 / 1008e9 ≈ 82 FLOP/byte
```

我们再计算一下 标准 attention（N=8192，d=128，FP32）所处的位置。
其中QKVO 各自约 4MB，需要读写一次；
两个中间矩阵S和P都是 `N×N`的 FP32 张量，各自268 MB，每个都需要写一次，再读回一次，总显存流量约为 1088 MB。


算一下。Q、K、V、O 各约 4 MB，各读/写一次；两个中间矩阵才是大头：S 和 P 各是 N×N 的 FP32 张量，各 268 MB，每个都要写一次再读回一次。总显存流量约 1088 MB。

它的算术量是 2 次矩阵乘：QKᵀ 是 `2·N²·d`，P·V 也是 `2·N²·d`，共 `4·N²·d ≈ 4·8192²·128 = 34.4 GFLOP`。

```
I = 34.4e9 / 1088e6 ≈ 32 FLOP/byte
```

很明显，标准 attention 落在 roofline 的带宽受限区。换句话说，它的算力利用率只有 `32/82 ≈ 39%`，剩下 61% 的 FLOPs 都在等显存。

同样的计算，但如果中间矩阵留在片上，K/V 块的重读又被 L2 缓存吸收，显存流量只剩 Q、K、V、O 各一次，约 16 MB。算术强度变成 `34.4e9 / 16e6 ≈ 2150 FLOP/byte`，远超 ridge 点，进入计算受限区，理论耗时 `34.4e9 / 82.6e12 ≈ 416 us`，和带宽受限差了约 2.6 倍。


> [!NOTE]
> 标准attention的瓶颈：`S` 和 `P` 这两个 N×N 的中间矩阵，本来算完就可以丢，却被迫经过 HBM 显存，产生开销。

## 2. FlashAttention 的优化核心：分块 + online softmax

经过上面的分析我们知道 attention 的核心开销就是优化中间矩阵，FlashAttention 的解法是**让 `S` 和 `P` 一直待在片上 SRAM**。


### 2.1. 分块计算
但是SRAM 很小（几十到几百 KB），没办法放下完整的中间矩阵，因此一个更加自然的思路就是把矩阵切片，分块来进行计算。

我们既然考虑到了分块计算，那究竟是怎么分块的呢？
attention 里输出的每一行都依赖Q的第 i 行，还有全部的KV。

为什么输出行只由对应的 Q 行决定？拿一个 N=2、d=3 的极小例子逐行计算（为排版省略 $1/\sqrt{d}$ 缩放，不影响行依赖结构）：

```math
Q = \begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \end{pmatrix}, \quad
K = \begin{pmatrix} 2 & 1 & 0 \\ 1 & 1 & 2 \end{pmatrix}, \quad
V = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 2 & 1 \end{pmatrix}
```

第一步，算 S = QKᵀ，观察每一行是怎么来的：

```math
S = \begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \end{pmatrix}
    \begin{pmatrix} 2 & 1 \\ 1 & 1 \\ 0 & 2 \end{pmatrix}
  = \begin{pmatrix} 2 & 5 \\ 1 & 3 \end{pmatrix}
```

| S 的行 | 逐元素计算 | 用到的 Q 行 |
|---|---|---|
| 第 0 行 | [1·2+0·1+2·0, 1·1+0·1+2·2] = [2, 5] | 只用 q₀ |
| 第 1 行 | [0·2+1·1+1·0, 0·1+1·1+1·2] = [1, 3] | 只用 q₁ |

S 的第 i 行只由 q_i（Q 的第 i 行）和全部 K 算出来，Q 的其他行不参与。

第二步，softmax 按行归一化，P 的第 i 行只由 S 的第 i 行决定：

```math
P = \text{softmax}(S, \text{dim}=-1) \approx \begin{pmatrix} 0.05 & 0.95 \\ 0.12 & 0.88 \end{pmatrix}
```

第三步，O = P·V，O 的第 i 行等于 P 的第 i 行乘 V：

```math
O = \begin{pmatrix} 0.05 & 0.95 \\ 0.12 & 0.88 \end{pmatrix}
    \begin{pmatrix} 1 & 0 & 1 \\ 0 & 2 & 1 \end{pmatrix}
  = \begin{pmatrix} 0.05 & 1.90 & 1.00 \\ 0.12 & 1.76 & 1.00 \end{pmatrix}
```

O 的第 0 行 = 0.05·v₀ + 0.95·v₁，系数 0.05/0.95 来自 P 的第 0 行，而 P 的第 0 行只来自 q₀；第 1 行同理只来自 q₁。所以 **O 的第 i 行从头到尾只用到了 Q 的第 i 行（q_i）和全部 K、V，Q 的其他行并没有参与**，每个 Q 行块都是可以独立算完的单元。


于是自然而然我们就把 Q 按行切成块。对每个 Q 块，遍历所有 K/V 块，在片上算完这一小块 attention 就累加进输出，`S`、`P` 就会只存在于 SRAM。

### 2.2. online softmax
> 参考我博客的softmax详解 [博客地址](https://dlog.com.cn)


另一个难点在 softmax。softmax 的分母需要整行的 max 和 sum，但我们是分块读的，读到一半时还不知道全行的 max。FlashAttention 用的是 **online softmax**，即维护一个 running max `m` 和 running sum `l`，每读一个新块，用新旧 max 的差去修正之前的累加结果。


核心代码如下：
```python
def flash_attention(Q, K, V, block):            # Q, K, V: [N, d]
    N, d = Q.shape
    O = torch.zeros_like(Q)
    for i in range(0, N, block):
        Qi = Q[i:i + block]                      # [block, d]
        m = torch.full((block, 1), -float('inf'))  # running max
        l = torch.zeros(block, 1)                # running sum
        Oi = torch.zeros(block, d)
        for j in range(0, N, block):
            Kj = K[j:j + block]                  # [block, d]
            Vj = V[j:j + block]
            Sij = Qi @ Kj.T / (d ** 0.5)         # [block, block]，只在片上
            m_new = torch.maximum(m, Sij.max(dim=-1, keepdim=True).values)
            # 用 exp(m - m_new) 把旧的 l 和 O 修正到新 max 尺度
            l = l * torch.exp(m - m_new) + torch.exp(Sij - m_new).sum(dim=-1, keepdim=True)
            Oi = Oi * torch.exp(m - m_new) + torch.exp(Sij - m_new) @ Vj
            m = m_new
        O[i:i + block] = Oi / l
    return O
```

网上讲 FlashAttention 大多画一堆分块图，其实上面这十几行就可以把核心说完。`Sij` 只有 `block × block` 大小（例如 128×128），SRAM 完全放得下；唯一需要全局信息的 softmax 归一化，用 `m` 和 `l` 两个 running 值搞定，把原本一次性计算的形态变成了滑动窗口或者是动态的感觉。

显存读写从 O(N²) 降到 O(N)，这就是 FlashAttention 加速的本质。
至于 FA2 则是在序列长度维度上再分块并行、FA3 针对 Hopper 架构做 warp specialization，都是在同一套分块 + online softmax 框架上的工程优化。

### 2.3 FA2 的优化：去掉跨 warp 归约

前面的 Tiling + online softmax 是 FA1 的主体框架。FA1 在 block 内部还会把 K 维度切给不同 warp，也就是 split-K。这样每个 warp 只计算 `S` 的一部分列。但 softmax 需要整行的 max 和 sum，所以每来一个新的 K 块，都需要把各 warp 的中间结果汇总到共享内存，做一次跨 warp 归约，并配合 `__syncthreads()` 同步。K 块越多，这类归约和同步越频繁。

FA2 的做法是改变 block 内部的分工方式：不再让不同 warp 分别处理 K 块，而是让每个 warp 固定负责一部分 Q 行，并独立遍历所有 K/V 块。这样每个 warp 都有自己的 `m`、`l`、`O`，中间状态不需要交给其他 warp，也就省掉了跨 warp 归约和同步。

FA1 的 block 内部仍然按 K 块分工：

```python
def fa1_block(Qi, K, V):
    # 一个 block 负责一批 Q 行
    m = -inf
    l = 0
    Oi = 0

    for j in range(0, N, Bc):
        Sij = Qi @ K[j:j+Bc].T

        # 不同 warp 只算出 S 的一部分列，
        # 这里需要先在共享内存中归约出整行 max
        m_new = max(m, Sij.max(dim=-1))  # 需要 __syncthreads()

        l = l * exp(m - m_new) + exp(Sij - m_new).sum(dim=-1)
        Oi = Oi * exp(m - m_new) + exp(Sij - m_new) @ V[j:j+Bc]
        m = m_new

    return Oi / l
```

FA2 则让每个 warp 自己负责一部分 Q 行：

```python
def fa2_warp(Qw, K, V):
    # Qw 是当前 warp 负责的那部分 Q 行
    m = -inf
    l = 0
    Ow = 0

    for j in range(0, N, Bc):
        s = Qw @ K[j:j+Bc].T

        m_new = max(m, s.max(dim=-1))
        l = l * exp(m - m_new) + exp(s - m_new).sum(dim=-1)
        Ow = Ow * exp(m - m_new) + exp(s - m_new) @ V[j:j+Bc]
        m = m_new

    return Ow / l
```

这里每个 warp 的 `m`、`l`、`Ow` 都只在 warp 内部更新，不需要写入共享内存给其他 warp 合并，也不需要 `__syncthreads()`。

下面用一个简化例子说明。假设 N=4，有两个 warp：warp0 负责 q0，warp1 负责 q1。两个 warp 各自完整遍历 4 个 K/V 块：

| K/V 块 j | warp0：s（q0） | warp0：(m, l) | warp1：s（q1） | warp1：(m, l) |
|---|---|---|---|---|
| j=0 | 1 | (1, 1.0000) | 2 | (2, 1.0000) |
| j=1 | 3 | (3, 1.1353) | 4 | (4, 1.1353) |
| j=2 | 0 | (3, 1.1851) | 1 | (4, 1.1851) |
| j=3 | 2 | (3, 1.5530) | 0 | (4, 1.2034) |
| 最终 O |  | (0.5930, 0.6760) |  | (0.1843, 0.8723) |

两个 warp 的中间状态完全独立：warp0 的 `m` 最终是 3，warp1 的 `m` 最终是 4，`l` 和 `O` 也按各自的 Q 行更新。因为不存在跨 warp 的中间结果合并，也就不需要额外的同步。相比 FA1，若同样处理 4 个 K/V 块，就可以省去 4 次跨 warp 归约和相应的同步开销。

FA2 的另一个收益是降低非矩阵乘操作的比例。FA1 中，softmax 相关的缩放和更新会随着 K 块推进反复发生；FA2 重新组织任务划分后，这部分开销更集中，矩阵乘在整体计算中的占比更高，Tensor Core 的利用率也更容易提升。

FA3和FA4等都是随着硬件架构加入了更新的一些细粒度操作，我们在这里按下不表。

## 3. flash-attn 库的函数分类

在使用 Dao-AILab 的 `flash-attn` 库时，你会发现它对外暴露了 7 个主要的 Python 接口。初看之下可能会觉得有些繁杂，但实际上，这 7 个函数底层调用的都是同一套高效的 FlashAttention CUDA kernel。它们之所以被拆分成不同的接口，主要是为了适配三种不同的现实需求：
- 输入张量在内存中是否打包（Packed）、
- Batch 内的序列是否等长（Varlen），
- 是否处于带有 KV Cache 的推理 Decode 阶段。

理解了这三个维度，就能很自然地掌握这些函数的使用场景。

| 函数 | 输入形状 | 用途 |
|---|---|---|
| `flash_attn_func` | Q/K/V 各 [B, S, H, D] | 最基础的定长接口 |
| `flash_attn_qkvpacked_func` | [B, S, 3, H, D] | QKV 打包成一个张量，省一次拆分 |
| `flash_attn_kvpacked_func` | [B, S, 2, H, D] | KV 打包，Q 单独传 |
| `flash_attn_varlen_func` | 变长 + cu_seqlens | 变长序列合批，避免 padding |
| `flash_attn_varlen_qkvpacked_func` | 变长 + QKV 打包 | 上两者的组合 |
| `flash_attn_varlen_kvpacked_func` | 变长 + KV 打包 | 上两者的组合 |
| `flash_attn_with_kvcache` | Q [B, 1, H, D] + 整页 KV | decode 专用，直接读 KV cache |

### 3.1. 基础定长接口
我们先从最基础的 `flash_attn_func` 看起。这是最标准的定长 Attention 接口，要求传入的 Q、K、V 是三个独立的张量，且形状均为 `[B, S, H, D]`（分别代表 Batch size、Sequence length、Head 数量和 Head dimension）。

这种接口适用于最规整的场景：Batch 内所有序列的长度完全一致，且模型在计算时已经把 Q、K、V 分别投影成了三个独立的张量。比如在进行模型训练，或者推理的 Prefill 阶段处理已经被 Pad 到相同长度的 Prompt 时，直接调用这个函数是最简单直接的。

### 3.2. Packed 系列
在很多 Transformer 变体或优化实现中，为了减少 Kernel Launch 次数和内存访问，Q、K、V 的线性投影往往是合并在一起计算的（即 `qkv_proj`）。这就导致投影出来的结果天然就是一个打包好的张量。

如果此时强行把这个打包的张量拆分成独立的 Q、K、V 再传给基础接口，不仅代码显得啰嗦，在某些非连续内存布局下还会触发额外的数据拷贝。为了解决这个问题，`flash-attn` 提供了 Packed 系列接口。

`flash_attn_qkvpacked_func` 接收一个形状为 `[B, S, 3, H, D]` 的单一 QKV 张量。这里的 `3` 就代表 Q、K、V 三份数据。如果你的模型直接输出了这种形状的张量，直接传给这个接口即可，省去了手动拆分的麻烦。

同理，`flash_attn_kvpacked_func` 适用于 Q 单独计算，而 K 和 V 合并投影的场景。它接收一个独立的 Q 张量 `[B, S, H, D]` 和一个打包的 KV 张量 `[B, S, 2, H, D]`。这在某些 Cross-Attention 结构，或者特定的 KV Cache 组织方式中非常常见。

### 3.3. Varlen 系列
在实际业务中，一个 Batch 里的序列长度往往参差不齐。如果使用基础的定长接口，就必须把所有序列 Pad 到 Batch 内的最大长度。这不仅浪费了显存，还会让 GPU 在大量的 Padding Token 上做无效的 Attention 计算。

Varlen（Variable Length）系列接口就是为了解决这个问题而生的。它打破了规整的 `[B, S]` 形状限制，将 Batch 内所有序列的 Token 拍平成一个一维的长序列，总长度为所有序列长度之和。此时，Q、K、V 的形状变成了 `[total_tokens, H, D]`。

既然 Token 被拍平了，Kernel 怎么知道哪些 Token 属于同一条序列，从而避免跨序列进行 Attention 计算呢？这就引入了 `cu_seqlens`（cumulative sequence lengths，累计序列长度）参数。

举个例子，假设 Batch 内有两条序列，长度分别是 3 和 4。拍平后的总 Token 数是 7。此时传入的 `cu_seqlens` 就是 `[0, 3, 7]`。它告诉 Kernel：第 0 条序列占据索引 `[0, 3)` 的位置，第 1 条序列占据索引 `[3, 7)` 的位置。配合 `max_seqlen` 参数，Kernel 就能精准地为每条序列独立计算 Attention，完全不需要 Padding。

基于这个核心思想，`flash-attn` 衍生出了三个 Varlen 接口：
- `flash_attn_varlen_func` 处理独立的 QKV；
- `flash_attn_varlen_qkvpacked_func` 处理打包的 QKV；
- `flash_attn_varlen_kvpacked_func` 处理打包的 KV。

它们分别对应了定长接口中的三种内存布局，只是加上了变长合批的能力。

### 3.4. Decode 阶段与 KV Cache
前面提到的所有接口，主要都服务于训练阶段或推理的 Prefill 阶段，也就是对一段完整的序列做 Attention。但在大模型推理的 Decode 阶段，情况发生了根本变化。

在 Decode 阶段，模型每次只生成一个新的 Token。这意味着当前的 Q 只有 1 个 Token（形状通常为 `[B, 1, H, D]`），而它需要去和历史上所有的 K 和 V 做 Attention。这些历史的 K 和 V 并不会每次都重新计算，而是被缓存在显存的 KV Cache 中。

如果依然使用普通接口，框架就需要每次把庞大的 KV Cache 拷贝或拼接成一个新的张量再传进去，这会给本就计算量极小的 Decode 阶段带来巨大的内存搬运开销。

`flash_attn_with_kvcache` 就是专门针对这个痛点设计的。它允许你直接把当前步的 Q 和指向 KV Cache 的指针（或 PageTable）传进去，Kernel 会直接去 Cache 中读取历史 KV 进行计算。这极大地减少了推理 Decode 阶段的内存读写，是提升大模型生成吞吐量的关键接口。

## 4. PagedAttention 讲解
上面说 `flash_attn_with_kvcache` 直接读 KV cache，但 KV cache 不是连续分配的。

flash-attn 的 `flash_attn_varlen_func` 和 `flash_attn_with_kvcache` 都接受 `block_table` 参数，这就是它支持 PagedAttention 的方式。

## 5. v3 实际调用的两个函数

v3 把 attention 拆成 prefill 和 decode 两个阶段，各用一个 FlashAttention 入口。

prefill（`flash_attn_varlen_func`）：

```python
o = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens_q,      # 变长 Q 的边界
    cu_seqlens_k=cu_seqlens_k,      # 变长 K 的边界
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    causal=True,
    block_table=block_tables,       # 续 chunk 时直接读分页 KV
)
```

prefill 时一条序列可能有多个 token 一次算完，多序列长度不等就用 `cu_seqlens` 合批。其中续 chunk（前缀缓存命中，`block_table` 非空）时 k/v 直接是整页池视图，内核按 block_table 寻址；首 chunk（无缓存）时 k/v 是本次算出的连续张量。

decode（`flash_attn_with_kvcache`）：

```python
o = flash_attn_with_kvcache(
    q.unsqueeze(1),                 # [B, 1, H, D]，每个序列一个新 token
    k_cache, v_cache,               # 整页池
    cache_seqlens=cache_seqlens,    # 每条序列当前 KV 长度
    block_table=block_tables,
    causal=True,
)
```

decode 每个序列只出一个 token，整批所有序列的 paged attention 在这个内核里一次算完，kernel 数不随 batch 增长。

v3 的 `Attention.forward` 本质上就是先写 KV 再选入口，精简后是这样：

```python
def attention(q, k, v, slot_mapping, is_prefill, ...):
    # 1. 把本步算出的 k/v 写进分页 KV cache
    store_kvcache(k, v, k_cache, v_cache, slot_mapping)

    # 2. 按阶段选 FlashAttention 入口
    if is_prefill:
        o = flash_attn_varlen_func(q, k_or_cache, v_or_cache, ..., block_table=...)
    else:
        o = flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens, block_table)
    return o
```

真实代码里它被注册成 `torch.library.custom_op("vllm::attention")`，这是为了让它能进 `torch.compile` 全图（Dynamo 追不进 FlashAttention 内核内部，注册成 opaque op 就当作黑盒跳过），这个机制本身不影响 attention 的语义。

## 6. GQA：内核原生处理，不展开 K/V

Qwen3-0.6B 是 16 个 query 头配 8 个 KV 头，即 GQA（Grouped Query Attention），一个 KV 头服务两个 query 头。

标准做法是把 K/V 从 [8, S, D] 广播成 [16, S, D] 再算，这会额外物化一份 2 倍大的 K/V。FlashAttention 内核原生支持 GQA，直接传 `num_qo_heads` 和 `num_kv_heads`，头映射在内核里完成，不物化中间张量。

v2 的 `_expand_gqa` 就是那个被省掉的物化操作：

```python
# v2：显式把 K/V 展开成 query 头数
k = k.unsqueeze(2).expand(b, hkv, num_kv_groups, s, d).reshape(b, num_heads, s, d)
```

## 7. v3 attention 的一条完整数据流

把前面拼起来，decode 阶段一层 attention 在 v3 里是这样走的：

```python
# 1. 投影并拆分出 Q / K / V
q, k, v = qkv_proj(hidden).split(qkv)

# 2. 对 Q / K 做 norm 和 RoPE
q = apply_rope(q_norm(q))
k = apply_rope(k_norm(k))

# 3. 将新的 K / V 写入 KV Cache
# 使用 Triton 单 kernel 完成写入
store_kvcache(
    k,
    v,
    k_cache,
    v_cache,
    slot_mapping,
)

# 4. 基于 KV Cache 做整批 Attention
# 同样是单 kernel 完成
attn_out = flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    cache_seqlens=cache_seqlens,
    block_table=block_table,
)

# 5. 输出投影
output = o_proj(attn_out)
```

#### prefill 阶段：同一份代码，靠 `is_prefill` 分支走不同入口

上面展示的是 decode 路径。prefill 阶段前面的步骤完全相同，唯一区别在 attention 这一步：`attention` op 里根据 `is_prefill` 和 `block_tables` 选择入口：

```python
# attention op 内部，decode 与 prefill 的入口选择
if is_prefill:
    # prefill：变长合批，varlen 接口
    if block_tables is not None:
        k, v = k_cache, v_cache           # 续 chunk：读前缀缓存里已算好的 KV
    else:
        k, v = k, v                       # 首 chunk：用本步刚算出的连续 KV
    o = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,        # 变长 Q 的边界
        cu_seqlens_k=cu_seqlens_k,        # 变长 K 的边界
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=True,
        block_table=block_tables,
    )
else:
    # decode：每个序列一个新 token，直接读整页 KV
    o = flash_attn_with_kvcache(
        q.unsqueeze(1),
        k_cache, v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_tables,
        causal=True,
    )
```

prefill 和 decode 在源码里其实是同一个 `attention` 函数、同一个 `forward`，靠 `attn.is_prefill` 这个布尔值分流。区别有三个：

1. **入口函数不同**：prefill 用 `flash_attn_varlen_func`（支持变长合批 + 前缀缓存续算），decode 用 `flash_attn_with_kvcache`（逐 token 直接读整页 KV）。
2. **KV 来源不同**：prefill 的首 chunk 直接使用本步刚算出的连续 k/v；续 chunk 和 decode 一样从 KV cache 里取。
3. **Q 的形态不同**：prefill 的 q 是 `[total_tokens, num_heads, head_dim]` 扁平变长；decode 则 `unsqueeze(1)` 成 `[B, 1, H, D]`。

写入 KV cache 这一步（`store_kvcache`）在两种阶段都会执行，prefill 一次写入一批新 token 的 KV，decode 每步只写每个序列新增的那 1 个 token。


## 8. 为什么没有自己手写 FlashAttention

核心算法（分块 + online softmax）用几十行代码就能讲清楚，但真正要在 4090 上跑出接近带宽极限的性能，还牵扯到：共享内存的 bank conflict 规避、warp 级并行调度、不同 block size 的 autotune、GQA / 变长 / 分页的组合处理。这些工程细节才是 FlashAttention 库真正难的地方。

所以 v3 的选择是直接调 flash-attn 的现成内核，把省下的精力放在 paged KV 布局、前缀缓存、CUDA Graph 这些系统层优化上。

后边会做一个面试专用版放在博客上。

> 一个实际感受：flash-attn 安装需要在本地编译，经常把 CPU 核跑满、连 ssh 都会卡。从上手体验来说 FlashInfer 更友好，这也是为什么 RoPE 和 SiLU 这两个轻量融合内核还留着 FlashInfer 的。
