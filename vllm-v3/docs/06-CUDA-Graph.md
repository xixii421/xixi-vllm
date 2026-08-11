# CUDA Graph Decode Runner 的设计与实现

> **导读**：在 LLM 的 Decode 阶段，模型每次仅处理一个 Token，计算量极小。然而，CPU 频繁向 GPU 下发计算指令（Kernel Launch）的开销，往往超过了计算本身的时间。本文将由整体到局部深度拆解 `CUDAGraphRunner` 模块的设计哲学、核心接口与工程亮点，带你掌握工业级推理引擎的底层优化精髓。

## 一、 为什么需要 CUDA Graph？

在传统的 PyTorch 动态图执行模式下，每一次前向传播都伴随着大量的 CPU-GPU 交互。我们可以用一个通俗的类比来理解：

- **传统模式**：就像一位**老板（CPU）** 指挥**工人（GPU）** 搬砖。老板需要依次下达指令：“去拿砖”、“走到指定位置”、“放下砖”、“回来报告”。在这个过程中，**老板下达指令的时间（Kernel Launch Overhead），往往比工人实际搬砖的时间还要长**。这导致 GPU 大量时间处于“等待指令”的空闲状态。
- **CUDA Graph 模式**：老板提前绘制了一张 **全自动施工图纸**，将一系列操作打包录制下来。后续每次需要搬砖时，老板只需下达一次指令，GPU 便会以最高效率、无停顿地完成所有计算，彻底消除中间的通信与调度延迟。

`CUDAGraphRunner` 正是基于这一理念，专为 Decode 阶段设计的计算图录制与重放引擎。

## 二、 整体架构设计

> 该模块的设计遵循的核心原则是：**静态内存分配，动态数据注入**。

由于 CUDA Graph 要求在录制（Capture）”和“重放（Replay）时，所有参与计算的 Tensor **内存地址必须绝对固定**，因此模块在架构上分为三个核心阶段：

1. **初始化**：在 GPU 上预先分配好所有输入、输出及中间状态的静态 Buffer，并建立 CPU 到 GPU 的高速数据通道。
2. **捕获（Capture）**：使用 Dummy（虚拟）数据对模型进行“热身”并录制计算图，生成不同 Batch Size 对应的静态执行图。
3. **重放（Replay）**：在真实的推理循环中，仅将最新的 Token ID 和动态的 KV Cache 索引快速覆盖到静态 Buffer 中，随后触发 Graph 执行。



## 三、 核心组件与接口剖析

下面我们由外到内，逐层拆解代码的核心实现与关键 API。

> [代码路径cuda_graph.py](../src/core/cuda_graph.py)



### 1. 初始化：构建静态内存与双缓冲机制 (`__init__`)

CUDA Graph 的核心约束在于内存地址与张量形状的绝对静态性。在计算图的捕获（Capture）与重放（Replay）阶段，所有参与计算的张量必须驻留在固定的内存地址，禁止任何动态内存分配。为满足此约束，系统在初始化阶段构建了以下底层数据结构：

```python
self.input_ids = torch.zeros(max_batch_size, dtype=torch.int64, device=device)
self.positions = torch.zeros(max_batch_size, dtype=torch.int64, device=device)
self.slot_mapping = torch.full((max_batch_size,), -1, dtype=torch.int32, device=device)
```

作为 CUDA Graph 的固定输入/输出接口。在 Replay 阶段，系统通过切片赋值（如 self.input_ids[:bs] = new_data）进行原地数据覆盖，确保张量内存地址不变，从而保证计算图能正确读取最新数据。

- input_ids 与 positions 采用 int64：严格对齐 PyTorch 原生 Embedding 层与 RoPE（旋转位置编码）算子的底层 CUDA 实现规范，避免隐式类型转换带来的性能损耗。
- slot_mapping 采用 int32：遵循 FlashInfer 等 PagedAttention 算子的索引数据类型标准。在满足最大显存寻址空间的前提下，将内存占用减半。

准备好地址之后，接下来来也要提供Host-Device的双缓冲机制：

```py
self._host_bufs = { bs: { "indptr": torch.zeros(..., pin_memory=True), ... } } # CPU 端
self._cg_bufs   = { bs: { "indptr": torch.zeros(..., device=device), ... } }   # GPU 端
```

这是为了调节动态业务逻辑与静态计算图的矛盾。Decode 阶段序列长度动态变化，导致 KV Cache 的物理块索引每步均不相同。

- _host_bufs（CPU 端）：负责动态数据的重组。利用 CPU 算力（如 NumPy）将变长的 block_tables 快速拼接为 FlashInfer 所需的 CSR 格式索引（indptr 与 indices）。
- _cg_bufs（GPU 端）：作为 CUDA Graph 内部固定的索引读取源。CPU 端重组完成后，将数据覆盖至此，供 Graph 重放时直接读取。

同时为了实现 Host-to-Device 的异步零拷贝，消除 PCIe 传输对主线程的阻塞，这里选择使用`pin_memory=True` 向操作系统申请页锁定内存，阻止系统将此部分物理内存交换至磁盘或移动其物理地址。这使得 GPU 的 DMA 控制器能够绕过 CPU，直接通过 PCIe 总线读取数据。配合 .copy_(..., non_blocking=True) 接口，数据传输与 CPU 的后续调度逻辑得以完全 Overlap，将通信延迟降至硬件物理极限。

> 测试脚本见 [non_blocking.py](../tests/single_test/non_blocking.py)

```
=================================================================
 CUDA Graph Non-Blocking 传输重叠 (Overlap) 性能测试
 数据大小: 300 MB | 迭代次数: 30 次 (取平均值)
=================================================================

测试结果对比 (相同工作负载):
-----------------------------------------------------------------
 [串行模式 (Blocking)]     :    87.64 ms / 次  (T_cpu + T_pcie)
 [并行模式 (Non-blocking)] :    48.59 ms / 次  (max(T_cpu, T_pcie))
-----------------------------------------------------------------
```



### 2. 图捕获机制：从动态执行到静态确定的映射 (`capture`)

`capture` 方法是 `CUDAGraphRunner` 的核心初始化逻辑。其核心逻辑是将包含复杂控制流与潜在动态内存分配的 PyTorch 模型前向传播过程，固化为一组**确定性的 GPU Kernel 调用序列**，从而在后续的 Decode 阶段彻底消除 CPU 的 Kernel Launch 开销。

为满足 CUDA Graph 对绝对静态性的严苛要求，该方法主要包含以下三个核心机制：

#### 2.1 环境净化与状态重置

在启动捕获流程前，代码首先执行环境清理：

```python
torch.cuda.synchronize(self.device)
torch.cuda.empty_cache()
self.slot_mapping.fill_(-1)
```

**底层逻辑**：`synchronize` 确保此前所有异步的 GPU 操作均已执行完毕；`empty_cache` 释放未被引用的显存碎片；`fill_(-1)` 则将静态 Buffer 重置为无效状态。这三步保证录制起点的显存状态纯净，避免历史残留数据干扰计算图的录制。

#### 2.2 降序捕获与显存池复用策略

```python
graph_pool = None
# 从大到小捕获：确保最大的 graph 分配显存池（pool），后续较小的 graph 复用该池
for bs in sorted(self.bs_list, reverse=True):
    # ... 捕获逻辑 ...
    if graph_pool is None:
        graph_pool = graph.pool()
```

**底层逻辑**：系统需要为多个离散的 Batch Size（如 1, 2, 4, 8, 16...）分别录制计算图。若按常规顺序录制，每个 Graph 都会独立向 CUDA 运行时申请显存池，极易导致严重的显存碎片化，甚至引发 OOM（Out of Memory）。
**优化策略**：采用**降序遍历**。首先为最大 Batch Size 录制 Graph，此时系统会分配一个容量最大的 `graph_pool`。随后录制较小 Batch Size 时，通过 `pool=graph_pool` 参数强制其复用该显存池。由于小图所需的显存必然包含在大图的分配范围内，此举在不牺牲覆盖率的前提下，将显存碎片化降至最低。

#### 2.3 两步走录制法

这是确保底层算子（如 FlashInfer）兼容 CUDA Graph 的关键设计。录制过程被严格拆分为两个独立的上下文阶段：

**阶段 A：Warmup Run**

```python
# 独立上下文，确保算子内存分配在 capture 前完成
with forward_context(...), attention_context(meta):
    self.outputs[:bs] = self.model(input_ids=..., positions=...)
```

高度优化的底层算子（如 FlashInfer 的 `BatchDecodeWithPagedKVCacheWrapper`）在首次执行时，通常会根据输入形状动态分配内部 Workspace。如果直接在 Graph 录制上下文中执行，这种动态分配行为会违反 CUDA Graph 的静态约束，导致录制失败或引入不可预测的开销。因此，必须先在不启用 Graph 的情况下执行一次真实的 Forward Pass，强制算子完成所有动态内存分配，使其状态进入就绪状态。

**阶段 B：Capture Run**

```python
# 独立上下文，图中只包含一次 forward
with forward_context(...), attention_context(meta):
    with torch.cuda.graph(graph, pool=graph_pool):
        self.outputs[:bs] = self.model(input_ids=..., positions=...)
```

在确认算子状态稳定后，进入 `torch.cuda.graph` 上下文管理器。此时，PyTorch **不会实际执行计算**，而是拦截并记录底层 CUDA Kernel 的启动顺序、参数指针及依赖关系。由于输入数据（`input_ids`, `positions`）指向的是初始化时预分配的静态 Buffer，录制下的计算图将永远绑定这些固定的内存地址。

#### 2.4 状态固化与收尾

```python
self._metas[bs] = meta
self.graphs[bs] = graph
# ... 循环结束后 ...
self.kv_cache_pool.zero_()
torch.cuda.synchronize(self.device)
self._captured = True
```

将录制完成的 `graph` 对象及其配套的 `AttentionMetadata`（包含指向静态 Buffer 的指针）按 Batch Size 索引保存至字典中。最后，对 `kv_cache_pool` 执行清零操作，防止 Warmup 阶段产生的脏数据污染后续真实的推理过程，并将 `_captured` 标志位置为 `True`，宣告捕获流程安全结束。

### 3. 动态索引重建与重放 (`replay` & `_replan_cg`)
虽然计算图是静态的，但每个序列的真实长度和对应的 KV Cache 物理块是动态变化的。如何在不破坏 Graph 静态性的前提下更新这些数据？

答案是：**在 CPU 端极速重组索引，然后异步覆盖到静态 Buffer 中。**

#### 3.1 CPU 端 CSR 格式索引极速重组
```python
# 1. 在 CPU 端 (NumPy) 极速拼接 FlashInfer 所需的 CSR 格式索引
indptr_np = host["indptr"].numpy()
idx_np = host["indices"].numpy()

acc = 0
indptr_np[0] = 0
for i in range(real_bs):
    nb = (seq_lens[i] + block - 1) // block  # 计算当前序列所需的物理块数
    idx_np[acc : acc + nb] = block_tables[i][:nb] # 填充物理块号
    acc += nb
    indptr_np[i + 1] = acc # 记录累积偏移量
```
FlashInfer 等高效注意力算子在底层依赖 CSR（Compressed Sparse Row，压缩稀疏行）格式来定位变长序列的 KV Cache。由于 Python 原生的列表操作存在较大的解释器开销，系统直接获取 Pinned Memory 的 NumPy 视图，利用其连续的内存布局进行极速的切片赋值。此过程完全在 CPU 端完成，避免了在 GPU 上执行复杂的动态索引拼接操作，从而消除了额外的 Kernel Launch 开销。

#### 3.2 Host-to-Device 异步非阻塞传输
```python
# 2. 异步拷贝到 GPU 静态 Buffer
bufs["indices"][:acc].copy_(host["indices"][:acc], non_blocking=self._non_blocking)
```
重组完成后，数据需从 CPU 传输至 GPU。由于 `host["indices"]` 在初始化时已被分配为 `pin_memory=True`，调用 `.copy_(..., non_blocking=True)` 可激活 GPU 的 DMA 控制器。DMA 将绕过 CPU，直接在后台通过 PCIe 总线将数据搬运至 GPU 端的静态 Buffer（`bufs["indices"]`）。

#### 3.3 算子元数据动态重绑定
```python
# 3. 通知 FlashInfer 更新内部状态
self.cg_wrappers[pbs].plan(
    indptr=host["indptr"],
    indices=bufs["indices"][:acc],
    last_page_len=host["last_page"],
    # ... 其他配置参数 ...
)
```
这是将动态数据注入静态图的最后一步。`wrapper.plan()` 是 FlashInfer 的核心接口，其作用并非重新编译或修改计算图，而是**更新算子内部用于读取索引的指针**。由于 `bufs["indices"]` 的内存地址在初始化后从未改变，`plan` 操作仅仅是让算子指向该固定地址中刚刚被异步覆盖的最新数据。

#### 3.4 静态图重放 (Replay)
```python
# 4. 重放 CUDA Graph
with forward_context(...), attention_context(meta):
    self.graphs[pbs].replay()
```
当静态 Buffer 中的数据被更新，且底层算子的指针被重定向后，调用 `graph.replay()`。此时，GPU 将以极低的延迟，严格按照捕获时记录的 Kernel 序列执行计算，直接读取最新的 `input_ids`、`positions` 以及刚刚注入的 `block_tables` 索引，输出正确的 Hidden States。

## 四、 总结

`CUDAGraphRunner` 模块是连接高层调度逻辑与底层硬件执行的桥梁。它通过 **“静态内存预分配 + 动态索引异步注入 + 计算图一键重放”** 的三段式设计，成功地将 Decode 阶段原本繁琐的 CPU-GPU 交互，压缩为一次极简的指令下发。

对于追求极致低延迟的实时系统，掌握并应用 CUDA Graph 及其配套的静态内存管理策略，是突破性能瓶颈的必经之路。

> **延伸阅读建议**：感兴趣的读者可进一步查阅 [PyTorch CUDA Graphs 官方文档](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs) 以及 [FlashInfer 项目源码](https://github.com/flashinfer-ai/flashinfer)，深入理解 PagedAttention 在图模式下的内存布局细节。

