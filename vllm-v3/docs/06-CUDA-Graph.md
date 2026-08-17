# CUDA Graph Decode Runner 的设计与实现

> **导读**：在 LLM 的 Decode 阶段，模型每次仅处理一个 Token，计算量极小。然而，CPU 频繁向 GPU 下发计算指令（Kernel Launch）的开销，往往超过了计算本身的时间。本文由整体到局部深度拆解 `CUDAGraphRunner` 模块的设计哲学、核心接口与工程亮点，带你掌握工业级推理引擎的底层优化精髓。

## 1. 为什么需要 CUDA Graph？

在传统的 PyTorch 动态图执行模式下，每一次前向传播都伴随着大量的 CPU-GPU 交互。我们可以用一个通俗的类比来理解：

- **传统模式**：就像一位**老板（CPU）** 指挥**工人（GPU）** 搬砖。老板需要依次下达指令：“去拿砖”、“走到指定位置”、“放下砖”、“回来报告”。在这个过程中，老板下达指令的时间，往往比工人实际搬砖的时间还要长。这导致 GPU 大量时间处于“等待指令”的空闲状态。
- **CUDA Graph 模式**：老板提前绘制了一张 **全自动施工图纸**，将一系列操作打包录制下来。后续每次需要搬砖时，老板只需下达一次指令，GPU 便会以最高效率、无停顿地完成所有计算，彻底消除中间的通信与调度延迟。

`CUDAGraphRunner` 正是基于这一理念，专为 Decode 阶段设计的计算图录制与重放引擎。

## 2. 整体架构设计

> 该模块的设计遵循的核心原则是：**静态内存分配，动态数据注入**。

由于 CUDA Graph 要求在录制（Capture）和重放（Replay）时，所有参与计算的 Tensor **内存地址必须绝对固定**，因此模块在架构上分为三个核心阶段：

1. **初始化**：在 GPU 上预先分配好所有输入、输出及中间状态的静态 Buffer。
2. **捕获（Capture）**：使用 Dummy（虚拟）数据对模型进行“热身”并录制计算图，生成不同 Batch Size 对应的静态执行图。
3. **重放（Replay）**：在真实的推理循环中，仅将最新的 Token ID 和动态的 KV Cache 索引快速覆盖到静态 Buffer 中，随后触发 Graph 执行。

> [代码路径 cuda_graph.py](../src/compute/cuda_graph.py)



## 3. 核心组件与接口剖析
### 3.1. 初始化

CUDA Graph 的核心约束在于内存地址与张量形状的静态性。v3 的 `CUDAGraphRunner` **直接复用** `Batch` **里预分配的常驻 GPU buffer 作为图的静态输入**：
```python
# data/batch.py —— 常驻 GPU buffer，decode / prefill / CUDA Graph 共用
self.input_ids    = torch.zeros(max_tokens, dtype=torch.int64, device=device)
self.positions    = torch.zeros(max_tokens, dtype=torch.int64, device=device)
self.slot_mapping = torch.full((max_tokens,), -1, dtype=torch.int32, device=device)
self.cache_seqlens = torch.zeros(max_bs, dtype=torch.int32, device=device)
self.block_tables = torch.zeros(max_bs, max_blocks, dtype=torch.int32, device=device)
```

- `input_ids` 与 `positions` 采用 int64：严格对齐 PyTorch 原生 Embedding 层与 RoPE 算子的底层 CUDA 实现规范。
- `slot_mapping` / `cache_seqlens` / `block_tables` 采用 int32：遵循 FlashAttention 等 PagedAttention 算子的索引数据类型标准，在满足最大显存寻址空间的前提下把内存占用减半。

`CUDAGraphRunner` 自己只额外分配一份输出 buffer：

```python
self.outputs = torch.zeros(max_batch_size, hidden_size, dtype=dtype, device=device)
```


### 3.2. 图捕获机制
`capture` 将模型前向固化为确定性的 GPU kernel 序列。

#### 环境净化与状态重置

```python
torch.cuda.synchronize(self.device)
torch.cuda.empty_cache()
self.batch.slot_mapping.fill_(-1)
```

保证录制起点的显存状态纯净，避免历史残留数据干扰录制。

#### 降序捕获与显存池复用

```python
graph_pool = None
for bs in sorted(self.bs_list, reverse=True):   # 从大到小
    # ... capture ...
    if graph_pool is None:
        graph_pool = graph.pool()               # 最大图分配 pool，小图复用
```

降序遍历：先录最大 Batch Size 分配 `graph_pool`，后续小图通过 `pool=graph_pool` 复用，避免显存碎片化。

#### 两步走录制法

这是确保底层算子（如 FlashAttention）兼容 CUDA Graph 的关键。录制拆成两个阶段：

**阶段 A：Warmup Run**（图外，强制算子完成动态内存分配）：

```python
with attention_context(self._meta):
    self.outputs[:bs] = self.model(self.batch.input_ids[:bs],
                                   self.batch.positions[:bs], attn)
```

**阶段 B：Capture Run**（图内，只记录 kernel 序列）：

```python
graph = torch.cuda.CUDAGraph()
with attention_context(self._meta):
    with torch.cuda.graph(graph, pool=self._graph_pool):
        self.outputs[:bs] = self.model(self.batch.input_ids[:bs],
                                       self.batch.positions[:bs], attn)
self.graphs[bs] = graph
```

由于输入指向初始化时预分配的静态 buffer，录制下的图永远绑定这些固定内存地址。

### 3. 重放
虽然图是静态的，但每个序列的真实长度与物理页是动态变化的。v3 的做法是把动态数据的注入放在 `Batch.build_decode`（`model_runner._run_decode` 里在 replay 之前调用）：

```python
# data/batch.py build_decode —— 每步把真实数据 copy_ 进常驻 buffer
self._cp(self.input_ids[:bs],    [s.last_token for s in seqs], torch.int64)
self._cp(self.positions[:bs],    [len(s) - 1 for s in seqs], torch.int64)
self._cp(self.slot_mapping[:bs], [self._decode_slot(s, block_size) for s in seqs], torch.int32)
self._cp(self.cache_seqlens[:bs], [len(s) for s in seqs], torch.int32)
self._fill_block_tables(seqs)      # 覆盖 block_tables[:bs, :]
```

`_cp` 走 `pin_memory` + `copy_(..., non_blocking=True)`，CPU 打包与 GPU 计算重叠。padding 区域保持捕获时的安全值（`slot=-1`、`seqlen=0`、`block=0`），重放无需额外清理。

之后重放只需按真实 batch 找最近的捕获尺寸：

```python
def replay(self, bs: int) -> torch.Tensor:
    pbs = self._pad_bs(bs)        # 找到 >= bs 的最近捕获尺寸
    self.graphs[pbs].replay()
    return self.outputs[:bs]

def _pad_bs(self, bs: int) -> int:
    for b in self.bs_list:
        if b >= bs:
            return b
    return bs
```

`bs_list` 由 `_make_bs_list` 生成：`[1, 2, 4, 8] + range(16, max_bs+1, 16)`，覆盖常见 batch 档位，`can_use(bs)` 判断是否命中。

## 四、总结

`CUDAGraphRunner` 模块是连接高层调度逻辑与底层硬件执行的桥梁。它通过 **“静态内存预分配 + 动态注入 + 计算图重放”** 的三段式设计，成功地将 Decode 阶段原本繁琐的 CPU-GPU 交互，压缩为一次极简的指令下发。

对于追求极致低延迟的实时系统，掌握并应用 CUDA Graph 及其配套的静态内存管理策略，是突破性能瓶颈的必经之路。

> **延伸阅读建议**：感兴趣的读者可进一步查阅 [PyTorch CUDA Graphs 官方文档](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs) 以及 [FlashAttention 项目源码](https://github.com/Dao-AILab/flash-attention)，深入理解 PagedAttention 在图模式下的内存布局细节。

