# Prefill 加速
> [!NOTE]
> 核心矛盾：prefill 的序列长度是动态的，CUDA Graph 却要求固定形状。
> v3 的最终方案：**torch.compile 全图编译**。三个无法被 Dynamo 追踪的算子注册为 custom op，使整个模型编译成单图。
> 
> 设计原则：**首 chunk 编译、续 chunk 走 FlashAttention paged eager**；两者作用于不同 chunk，互不冲突。

## 1. 为什么 prefill 不能直接用CUDA Graph？

decode 每次只处理一条序列的一个 token，batch 形状离散且可枚举（1/2/4/8/…），可以用少量固定形状的静态图覆盖。prefill 则完全不同：

- 序列长度任意，`max_seqlen_q` / `cu_seqlens` 每批都在变；
- 一个 batch 内多个序列长度各不相同；
- 前缀命中时还要区分**首 chunk（无缓存）**与**续 chunk（有缓存）**。

如果硬要录图，只能像之前的版本 piecewise 方案那样把 prefill 切成若干 token 尺寸网格，每个尺寸录一套图。

但是这带来两类结构性缺陷（见下），最终被放弃。

### piecewise CUDA Graph 的两个结构缺陷（已放弃）

早期 v3 对齐官方 vLLM 的 piecewise 思路：attention 摘出 eager、其余层按 token 尺寸网格进图。实测发现两个问题：

1. **静态 buffer copy**：每次重放都要把整批 `input_ids`/`positions`/`slot_mapping` 拷进固定地址的静态 buffer，再回读输出，copy 次数反超 eager（profiler 实测 eager 168 次 copy → piecewise 267 次）。
2. **attention 摘出**：attention 无法进 torch 图，只能在图外 eager 单独跑，图内外各走一遍 kernel 序列，kernel 总数不降反升（646 → 784）。

## 2. 最终方案：torch.compile 全图编译
torch.compile 在编译期分析整张计算图，把相邻的逐元素算子融合成一个 kernel，并生成优化后的底层代码。原本几百次 kernel 发射缩减到几十次，发射开销和中间张量的显存往返同时减少。Inductor 还支持符号形状，动态长度一次编译即可覆盖，无需像 CUDA Graph 那样枚举形状。

CUDA Graph 录制已定型的发射序列，必须固定形状；torch.compile 重新生成算子更少、更高效的代码，天然适配变长。v3 的 decode 用前者，prefill 用后者。

但是全图编译的前提是 Dynamo 能追踪整张计算图，而 FlashAttention 与 FlashInfer 的 RoPE、SiLU 内核无法被追踪。

Dynamo 是 torch.compile 的前端，它逐行执行模型的 Python 代码，把遇到的每个算子记录成计算图上的一个节点。要成功记录，Dynamo 必须认识这个算子：知道它的输入输出签名，知道给定输入形状后输出形状如何推导，知道它能否安全放进图中。普通 PyTorch 算子如 `torch.add`、`F.rms_norm`、`nn.Linear` 都自带完整的元数据描述，Dynamo 天然认识它们，能一路追踪到底。

FlashAttention 与 FlashInfer 的 RoPE、SiLU 内核不满足这个前提。它们是独立编译好的 CUDA 内核，对外只是一次黑盒调用，内部没有以 PyTorch 算子形式存在，Dynamo 看不到任何可记录的中间运算；同时它们也没有配套的形状元数据，Dynamo 无法推断输出形状。面对这种算子，Dynamo 只能放弃追踪，在调用点处把模型切断：调用点之前的算子编成一段图，之后的算子编成另一段，切断处的运算退回 eager 执行。这个切断行为就是 graph break。

graph break 的代价很大。一次切断意味着编译出的图被拆成多段，段与段之间失去算子融合的机会，中间结果要完整落回显存，CPU 也重新介入调度。如果图中出现多个断点，编译收益基本被抵消，甚至退化成近似 eager。因此要让整个 prefill 编成一张完整的图，就必须把这三个内核包装成 Dynamo 认识的算子。

因此解决方案是把这三个算子注册为 opaque custom op，让编译器将它们视为黑盒。



一条 prefill 请求的 prompt 在 KV cache 里可能已经缓存了一部分，缓存可能来自上次推理留下的前缀，也可能来自别的请求共享的前缀。

而没有缓存的那一段，需要从零计算并写入 KV cache，称为首 chunk；已经存在缓存的那一段，KV 不必重算，只需要把注意力指到已有缓存上，称为续 chunk。

这个划分逐请求发生。没有任何前缀命中时，整条 prompt 都是首 chunk；命中全部前缀时只剩最后的续 chunk；部分命中时则两者兼有，前段是续 chunk、后段是首 chunk。

并发场景下，同一批请求各自独立地做这个划分。批内可能一部分请求完全没有命中，一部分请求部分命中，甚至命中比例各不相同。调度器为每个请求单独记录 `num_cached_tokens`，各自决定哪些 token 属于首 chunk。这也是为什么编译路径要求批内所有请求同时满足 `num_cached_tokens == 0`，只要有一个请求带缓存，整个 prefill 批次就走 eager。

再看两个 chunk 各自适合的执行方式。

首 chunk 要处理全部 token，形状多变，每批序列长度都不同，是 prefill 耗时的主体，也是算子融合收益最大的地方，值得交给 torch.compile。

续 chunk 走 FlashAttention paged eager 更合适，原因有二。一是它带着 `block_tables` 和 `cu_seqlens_k` 这两个缓存相关的分支，与编译图固定的纯 tensor 签名不匹配，硬塞进图会让图变复杂，而这些都是每批动态的元数据。二是续 chunk 的省算收益来自 paged attention 本身：KV 已经被缓存，注意力直接读缓存页，跳过重复的 KV 计算与写入，这个收益与编译无关。所以续 chunk 保持 eager，把精力省给首 chunk。

两者作用于不同的 chunk，互不冲突。

## 3. 实测

详见 [02-实测证据](./02-实测证据.md) 2.1 节，要点摘录：

| 指标 | eager | torch.compile | 变化 |
| --- | ---: | ---: | ---: |
| wall（L=512） | 23.51 ms | 5.98 ms | **3.93x** |
| kernel 数 | 646 | 537 | −109 |
| copy 次数 | 168 | 116 | −52 |

单请求 TTFT（短 prompt 收益大、长 prompt 收敛）：

| prompt len | eager (ms) | compile (ms) | 加速比 |
| --- | ---: | ---: | ---: |
| 128 | 17.74 | 4.93 | **3.60x** |
| 256 | 17.98 | 6.63 | **2.71x** |
| 512 | 18.08 | 10.84 | **1.67x** |
| 2048 | 43.58 | 42.54 | **1.02x** |

- 短 prompt 收益大（发射开销主导，3.6x）；长 prompt 收敛 ~1.0x（GPU-bound 物理边界）。

| 场景 | 耗时 |
| --- | ---: |
| 冷缓存编译（inductor 首次） | ~25 s |
| 热缓存编译（磁盘缓存命中） | ~10 s |
| 启动期预热（首次调用触发编译） | 计入引擎 Init，对首个请求的耗时没有影响 |

编译发生在引擎初始化阶段。`_build_compiled_prefill` 在构造时就用占位输入调用一次编译好的函数，把首次编译的耗时算在引擎启动里。引擎就绪后，每次请求直接命中编译产物，请求自身不再承担任何编译时间。`compile_dynamic` 控制形状策略：

- `True` = 符号形状一次编译覆盖任意长度（在线变长服务首选，任意长度不重编译）；
- `False` = 固定形状重放更快，但新长度会重编译（~2~12s，仅离线固定长度）。

## 4. 数值正确性

编译融合会改变 bf16 浮点运算顺序，eager 与 compile **不再是 bit 级一致**。实测（引擎强制 `allow_bf16_reduced_precision_reduction=False`，即 GEMM fp32 累加）：

- `max |eager − compiled| = 1.125`；
- greedy 逐 token 一致率 55/63 = 87.3%，分叉均为 near-tie argmax 翻转，属 bf16 固有数值扰动，无系统性偏差。



## 5. 知识点总结

1. CUDA Graph 要求固定形状，prefill 的动态长度使其无法直接录图；torch.compile 用符号形状（`dynamic=True`）天然适配变长。
2. 把不可 trace 的算子（attention / silu_and_mul / rotary_embedding）注册为 custom op 黑盒，是实现 fullgraph 的关键。
3. 编译只覆盖首 chunk；续 chunk 走 FlashAttention paged eager。
4. compile 消除的是发射开销而非计算，收益集中在 CPU-bound 短 prompt。
5. 代价是 eager/compile 不再 bit 一致（~1 ULP bf16 扰动），属数值保真 vs 性能的取舍。
