# 10 · Prefill CUDA Graph：piecewise

> 对应 [04-CUDA-Graph](./04-CUDA-Graph.md)（decode CG）。这一份专门讲 **prefill 怎么用 CG**。
> 核心矛盾：prefill 的序列长度是动态的，CUDA Graph 却要求固定形状。
> v3 对齐官方 vLLM 的 **PIECEWISE**：attention 摘出 eager，其余进图。
> 设计原则：**没有测速判据；网格 cap 由 CG 收益边界决定（2048），超过 cap 的批次
> 回退 eager，不 pad 进大图做负收益计算**。

## 一、为什么 prefill 也能用 CG？

### 1.1 prefill 的瓶颈是 CPU 发射地板

eager prefill 单步的 wall 时间里，GPU 计算只占一小部分，其余是 **数百次 kernel 发射的
CPU 串行化**（GPU 在等 CPU）。这正是 CUDA Graph 最擅长消除的开销。

短 prompt 时发射开销主导 → CG 收益最大（实测 ~2.5x）。长文本 prefill 是 GPU-bound
（GPU 打满），CG 消除发射开销的收益趋近于零——这是物理边界，任何实现都一样。

### 1.2 官方 vLLM 的做法（对齐目标）

官方 V1 的 piecewise 编译：

1. 在 **attention 边界 split** forward 成多段；
2. 非 attention 段（embed / norm / qkv_proj / rope / o_proj / mlp）按捕获尺寸捕获
   成 CUDA Graph；
3. attention 段保持 eager，首 chunk 用 FA2 varlen，续 chunk 用 FlashInfer paged；
4. token 维用少量捕获尺寸离散化，运行时 pad 到最近尺寸。

> **v3 与官方的一处刻意差异**：官方用 torch.compile（Dynamo symbolic shape）把网格
> 铺到调度批次上限；v3 复用 layer 算子（不 compile，理由见 2.6），网格上限由
> **CG 收益边界**决定而非调度器——超过收益边界的批次（GPU-bound 大 batch）回退
> eager，避免 pad GEMM + 层间拷贝的负收益。

### 1.3 v3 的解法：token 尺寸网格 + 复用 layer 算子

v3 无 Dynamo 的 symbolic-shape splitting，用「token 尺寸网格」离散化一次前向的总
token 数 `T`。每个尺寸捕获 `N+1` 张 piece 图（`N`=层数）：

```
piece_0 : embed + 层0 pre（input_norm + qkv + qk_norm + rope）→ q0,k0,v0
[eager] : 层0 attention（FA2 varlen / FlashInfer paged）→ attn_out_0
piece_1 : 层0 post（o_proj + post_norm + mlp）+ 层1 pre → q1,k1,v1
[eager] : 层1 attention
  ...
piece_N : 层N-1 post + final_norm → outputs
```

运行时决策**只有一个**：取 `>= 实际 token 数` 的最小捕获尺寸，pad 进图。

---

## 二、实现要点（`core/prefill_cuda_graph.py`）

### 2.1 三处关键设计（对比旧版）

| | 旧版 | 新版 |
| --- | --- | --- |
| 用不用图 | `can_use`：捕获时实测 wall，慢于 eager 就整桶禁用 | **永远用图**，`covers()` 只查 token 数是否 ≤ 网格上限 |
| 覆盖范围 | 硬顶 `_MAX_CG_TOKENS=1536`，超长回退 eager | **网格 cap = CG 收益边界（4090 上 2048）**，≤cap 必进图、>cap 回退 eager（GPU-bound 区间不进图） |
| 网格来源 | 由调度器 `max_num_batched_tokens` 驱动 | **固定网格** `(256,512,768,1024,1536,2048)`，与调度器解耦（调度仍可用 8192） |
| 续 chunk | 只加速首 chunk，续 chunk 走 eager | 续 chunk（前缀命中）也走同一套图，attention 换 FlashInfer paged |

### 2.2 一份 max-T 静态 buffer，所有尺寸共用切片

不再每桶一套 buffer：所有尺寸的图复用同一份 `max_tokens` 静态 buffer 的 `[:T]` 切片。
图数量、显存占用与「token 桶」时代脱钩，显存只随最大尺寸线性增长。网格 cap=2048 时
静态 buffer 约 29 MB，整个 prefill CG 足迹（workspace + 静态 buffer + graph pool）
实测仅 ~0.11~0.13 GiB。

### 2.3 复用 layer 算子（数值正确性关键）

piece 的计算**直接复用模型层的 `pre_forward` / `post_forward`**（cuBLAS GEMM +
flashinfer silu_and_mul + triton rope + F.rms_norm），与 eager 路径算子**完全同源**。
「真实 token 数 == 捕获尺寸」时 piecewise 与 eager **bit 一致**。

### 2.4 图 key 用 `(T, piece_idx)`

不同尺寸的 piece 图独立（共享 graph pool 只复用临时激活显存），不能用 piece_idx
单键——否则降序捕获时后捕获的尺寸会覆盖先捕获的图。

### 2.5 数值说明（两个独立差异源）

1. **GEMM M 敏感（固有，非 bug）**：token 尺寸网格把批次 pad 到捕获尺寸 `T`，
   cuBLAS 的 bf16 GEMM 对 M 维度敏感（~0.004），叠加 Qwen3 极小的 k 通道 rms
   （~0.007）经 k_norm 放大 ~140x，使有 padding 的 batch 在极小概率下与 eager
   分叉。这是 bf16 推理中「batch 大小变化改变输出」的固有现象（continuous
   batching 里本就存在）。精确尺寸（无 pad）bit 一致；同一 engine 内两次调用
   不存在该差异，差异只出现在「eager / CG 两个独立进程」的跨进程对比中。

2. **torch.compile 融合（已放弃，见下）**：compile 融合会额外改变数值，经实测无法与
   eager 的 cuBLAS 分开调用 bit 一致。

### 2.6 为什么放弃 torch.compile（实验结论）

曾尝试对齐官方「torch.compile 融合非 attention 段」，但实测在 bf16 下无法同时满足
「数值正确 + 性能收益」：

| compile 策略 | 数值（无 pad vs eager） | 性能 |
| --- | --- | --- |
| `max-autotune` 融合 | diff=0.5（分叉） | 有收益 |
| `default` + 可 trace rms_norm | diff=2.5（完整模型，仍分叉） | 有收益 |
| 全 `@torch.compiler.disable`（silu/rope/rms_norm） | **diff=0.0（bit 一致）** | **0.56x（更慢）** |

根因：bf16 下算子融合让中间结果不 round 回 bf16，改变数值；要 bit 一致只能禁用所有
融合，但 graph break 使 compile 反而更慢。故放弃 compile，复用 layer 算子——这是
「数值正确 + 性能最优」的唯一解。

---

## 三、实测（RTX 4090 D）

### 3.1 单请求 TTFT（`bench_prefill_cg.py`，enforce_eager 隔离 prefill CG 变量）

| prompt len | eager (ms) | prefill-CG (ms) | CG 加速 |
| --- | ---: | ---: | ---: |
| 128 | 17.22 | 5.44 | **3.17x** |
| 256 | 18.92 | 5.86 | **3.23x** |
| 512 | 17.03 | 7.12 | **2.39x** |
| 1024 | 16.34 | 11.50 | **1.42x** |
| 1536 | 18.50 | 16.75 | **1.10x** |
| 2048 | 25.14 | 23.63 | **1.06x** |
| 2560 | 31.72 | 29.53 | 1.07x（>cap 回退 eager） |
| 3072 | 37.61 | 36.76 | 1.02x（>cap 回退 eager） |

- 短 prompt 收益大（发射开销主导）；长 prompt 收敛到 ~1.0x（GPU-bound 物理边界）。
- **>2048 回退 eager**：2560/3072 的「CG」配置与 eager 同路径，比值 ≈1.0 只余噪声——
  超收益边界的批次不再 pad 进大图，这是与旧版「任意规模都 pad 进图」的本质区别。

### 3.2 输出正确性（`bench_verify_output.py`，greedy 逐 token）

| 场景 | 结果 |
| --- | --- |
| 精确网格尺寸（256/512/1024/1536/2048，无 pad） | **bit 一致** |
| 网格边界邻近（255/257/511/513/700/1023/1535/2049） | 一致（3/3） |
| 超网格（2049/3072/4096，回退 eager） | **与 eager 完全一致**（同代码路径） |
| 大并发（32 请求 sum=1463） | 一致 |
| 中大批（sum 600/1000/1413） | 一致 |
| 重 pad 小 prompt（1/7/37/100→256、257/300→512） | 少量分叉（bf16 GEMM M 敏感固有，见 2.5） |

总一致率 74/83（89.2%），分叉全部集中在重 pad 小 prompt 的早期 token（1~16），
是 2.5 节所述 bf16 固有现象，不是路径错误。

### 3.3 并发场景（`bench_prefill_cg_concurrent.py`）

见 [02-实测证据](./02-实测证据.md) 2.1 节。

**知识点总结**：
1. CUDA Graph 消除的是发射开销不是计算；收益集中在 CPU-bound 区间。
2. 动态形状用 token 尺寸网格离散化；**网格 cap = CG 收益边界（2048），超过 cap 的
   GPU-bound 大 batch 回退 eager**，避免 pad 负收益——网格与调度批次上限解耦。
3. 复用 layer 算子保证 eager/piecewise 数值同源（精确尺寸 bit 一致）。
4. 续 chunk 也进图（attention 换 paged），长 prompt 每一片都加速。
5. torch.compile 在 bf16 下无法「数值正确 + 性能收益」两全（融合改变数值，禁用融合更慢）。
6. prefill CG 显存足迹按实际捕获量预留（`Engine._probe_cg_footprint`），不依赖固定预留。
