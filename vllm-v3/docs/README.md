# vllm-v3 文档

**核心叙事：v2 调度已经对了，慢在算子；v3 专治这些性能瓶颈。**

控制面（CB / chunked / PD mix）与 v2 **同构**；相对 v2 的收益几乎全部来自 **数据面与计算面的换血**。

## 必读（按顺序）

| # | 文档 | 读什么 |
| --- | --- | --- |
| 1 | [00-优化总览](./00-优化总览.md) | **完整优化线路图**（从这里开始） |
| 2 | [01-v2性能瓶颈](./01-v2性能瓶颈.md) | v2 慢在哪（逐项拆解） |
| 3 | [02-实测证据](./02-实测证据.md) | 数字证明瓶颈被打掉（算子 → 机制 → 端到端三层） |

## 专题（按优化点深入）

| 文档 | 内容 | 知识点 |
| --- | --- | --- |
| [04-Attention](./04-Attention.md) | Attention 后端换血 | FlashAttention paged、KV 写 |
| [05-算子融合](./05-算子融合.md) | GEMM 融合 + FlashInfer RoPE/SiLU | 算子融合、`F.rms_norm` |
| [06-CUDA-Graph](./06-CUDA-Graph.md) | Decode CUDA Graph 设计与实现 | 图捕获/重放、静态内存、非阻塞传输 |
| [07-Prefill-torch.compile](./07-Prefill-torch.compile.md) | Prefill torch.compile 全图编译 | 动态形状、custom op、fullgraph |
| [08-显存优化](./08-显存优化.md) | **CG / compile 显存治理** | 显存分解、graph pool、共享安全判据 |
| [09-输入管线](./09-输入管线.md) | 输入数据管线 | cu_seqlens、slot_mapping、pin_memory |
| [10-Sampler](./10-Sampler.md) | 采样器 | 整批采样、Gumbel-max、同步开销 |
| [11-模型加载](./11-模型加载.md) | 快速权重加载 | 分片读取、dtype 直落 |

> v2 文档自己写了：*「涉及到算子的优化都没有做」*。
> v3 文档要回答的是：**那些没做的算子具体卡在哪，以及换掉之后快多少。**
> 以及：**CG / compile 带来的显存开销是多少，如何共享/取舍**（[08](./08-显存优化.md)）。
