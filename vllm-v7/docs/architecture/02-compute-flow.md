# 计算流（Compute Flow）

计算流回答：

> 拿到「本步要算的序列 + 是否 prefill」之后，**GPU 上如何变成新 token**？

它**不**维护 waiting/running，**不**决定下一个请求是谁。入口来自 execute（或 LocalExecutor 直接调用），出口是 `token_ids`（及可选 mRoPE 快照）。

---

## 1. 总览图（从桥到 kernel）

```mermaid
flowchart TB
  subgraph Bridge["桥 · 仍属控制侧发起"]
    SUB["Executor.submit_execute_model<br/>或 LocalExecutor 同步调用"]
    RPC["ZMQ/Pipe pickle · multiproc 时"]
  end

  subgraph WorkerLayer["Worker · 薄胶水"]
    ENV["NCCL / TP / PP 环境已就绪"]
    PPR["非 first PP stage:<br/>recv IntermediateTensors"]
    RUN["ModelRunner.run"]
    PPS["非 last PP stage:<br/>send IntermediateTensors · 不采样"]
    ENV --> PPR --> RUN --> PPS
  end

  subgraph Runner["runner/ · 单卡一步"]
    PRE["prepare_step<br/>tensors / positions / sampling 元数据"]
    VIS["vision encode_and_merge<br/>仅 VL + prefill"]
    CG{"decode 且 CUDA Graph 可用?"}
    FWD_CG["forward_cuda_graph · replay"]
    FWD_E["forward_eager · model.forward"]
    META["必要时 build_attn_meta"]
    SAMP["sample_from_hidden"]
    PRE --> VIS --> CG
    CG -->|是| FWD_CG --> SAMP
    CG -->|否| META --> FWD_E --> SAMP
  end

  subgraph Stack["models → layers → kernels"]
    M["Qwen3 / VL / MoE Module"]
    L["Attention · Linear · MoE · Norm · RoPE"]
    K["FlashInfer / cuBLAS / NCCL collectives"]
    M --> L --> K
  end

  SUB --> RPC --> WorkerLayer
  SUB -.->|ws==1 无 RPC| RUN
  RUN --> PRE
  FWD_CG --> M
  FWD_E --> M
  SAMP -->|"token_ids + mrope_updates"| OUT["回传 Engine"]
```

---

## 2. `ModelRunner.run` 逐步图

这是计算流的心脏。

```mermaid
flowchart TB
  IN["run(seqs, is_prefill, intermediate_tensors?)"] --> E0{"seqs 为空?"}
  E0 -->|是| E1["PP last: 空 token 列表<br/>否则空 IntermediateTensors"]
  E0 -->|否| C0["判断 cg_likely:<br/>decode ∧ graph 存在 ∧ batch 命中"]

  C0 --> P0["prepare_step"]
  P0 --> P1["拼 input_ids / positions<br/>block_tables / slot mapping<br/>采样温度等"]
  P1 --> P2["VL: 当前 window 重叠图像<br/>encode → merge 进 embeds"]
  P2 --> P3["CG 路径可跳过完整 attn_meta"]

  P3 --> U{"use_cg?"}
  U -->|是| G1["cuda_graph_runner.replay"]
  U -->|否| A1{"attn_meta 已有?"}
  A1 -->|否| A2["build_attn_meta · FlashInfer"]
  A1 -->|是| A3["forward_eager"]
  A2 --> A3

  G1 --> H["hidden 或 IntermediateTensors"]
  A3 --> H
  H --> PP{"是 PP 中间产物?"}
  PP -->|是| RET_I["返回 IntermediateTensors<br/>不采样"]
  PP -->|否| S1["sample_from_hidden"]
  S1 --> RET_T["返回 token_ids list<br/>+ mrope_updates"]
```

### 子阶段职责

| 阶段 | 包/文件 | 做什么 |
|------|---------|--------|
| 准备 | `step_prepare.py` | CPU/GPU 张量、页表、是否采样 |
| 视觉 | `runtime/vision/` | 仅 prefill；写 encoder cache |
| 注意力元数据 | FlashInfer wrapper | paged KV 的 indptr / indices |
| 前向 | `models/*` + `layers/*` | 真正 matmul / attn / moe |
| 图捕获 | `cuda_graph/decode.py` | decode 定长 batch 回放 |
| 采样 | `sampler.py` | logits → token id |

---

## 3. 多卡时计算流如何「长」在并行上

计算流内部才出现 NCCL；调度流仍然只看到一份逻辑 `seqs`。

```mermaid
flowchart TB
  subgraph EngineView["Engine 视角 · 逻辑一步"]
    L1["一份 seqs + is_prefill"]
    L2["一份 token_ids 结果"]
  end

  subgraph TP["Tensor Parallel · 同 PP stage"]
    T0["各 rank 持有权重分片"]
    T1["Layer 内 all-reduce / all-gather"]
    T0 --> T1
  end

  subgraph PP["Pipeline Parallel · 跨 stage"]
    P0["Stage0: embed + 前几层"]
    P1["Stagek: recv → 中间层 → send"]
    P2["StageLast: 后几层 → logits → sample"]
    P0 --> P1 --> P2
  end

  subgraph MoE["MoE · cutlass + TP"]
    E0["路由 top-k"]
    E1["cutlass_fused_moe (local shard)"]
    E2["TP all-reduce"]
    E0 --> E1 --> E2
  end

  L1 --> TP
  L1 --> PP
  TP --> MoE
  P2 --> L2
```

Worker 级伪代码（与实现一致）：

```text
if not first_pp_rank:
    intermediate = NCCL_recv(prev)
output, mrope = ModelRunner.run(seqs, is_prefill, intermediate)
if not last_pp_rank:
    NCCL_send(output, next)
    return (None, mrope)          # 中间 stage 不产生 token
else:
    return (token_ids, mrope)     # 仅 last stage 采样
```

Executor 侧再按 reply 策略 merge（通常 driver / 指定 rank 回包）。

---

## 4. 单卡退化（教学默认路径）

```mermaid
flowchart LR
  E["Engine"] --> L["LocalExecutor"]
  L -->|"同一进程函数调用"| M["ModelRunner.run"]
  M --> T["token_ids"]
  T --> E
```

- **没有** ZMQ，**没有** 多进程 spawn；  
- 仍走完整 prepare → forward → sample；  
- 仍可能 init 单卡 process group（统一代码路径的税）。  

讲解计算流时，**先画这张，再展开 Worker/NCCL**。

---

## 5. VL / MoE / CG 挂在计算流哪里

```mermaid
flowchart TB
  PRE["prepare_step"] --> VL{"is_multimodal ∧ prefill?"}
  VL -->|是| V1["vision.encoder_runner<br/>layers.vision ViT"]
  VL -->|否| CG
  V1 --> CG{"decode ∧ CG?"}
  CG -->|是| C1["cuda_graph.decode replay"]
  CG -->|否| F1["eager forward"]
  C1 --> MOE
  F1 --> MOE{"层内是 MoE block?"}
  MOE -->|是| M1["routing → experts → dispatch<br/>fused 或 fallback"]
  MOE -->|否| DENSE["dense MLP"]
  M1 --> OUT["继续后续层 / 采样"]
  DENSE --> OUT
```

设计约束：

- **VL**：`input/` 只做 CPU 特征；GPU encode 必须在计算流。  
- **CG**：主要加速 decode；prefill / ViT 另算。  
- **MoE**：Scheduler 无感；backend 选择属于计算面配置（当前 SM80 fused 有已知坑，见性能报告）。

---

## 6. 计算流应如何设计（分析）

### 已做对的

| 原则 | v7 体现 |
|------|---------|
| 单入口一步语义 | 只有 `ModelRunner.run` |
| 准备与前向分离 | `prepare_step` vs `forward_*` |
| CG 可旁路 | miss 时只补 attn_meta，不整步重来 |
| PP 采样位置正确 | 仅 last stage sample |
| 视觉与文本运行时拆分 | `vision/` 注入，不把 HF processor 塞进 Worker |
| 桥可替换 | Local / ZMQ / Pipe 不改变 run 契约 |

### 设计时注意的边界

1. **计算流输入应是「已调度快照」**  
   使用本步的 `num_scheduled_tokens` / `block_table`，不要回头改 Scheduler 队列。

2. **控制消息与张量消息分离**  
   token id 可以走 RPC；hidden 必须走 NCCL。中间 PP stage 回 `None` tokens，避免假采样。

3. **图捕获与动态形状**  
   CG 只覆盖 decode 固定 batch 集合；动态 prefill 走 eager——这是正确的能力边界，不是缺陷。

4. **状态回写要显式**  
   mRoPE 等用 `MRopeStateUpdate` 快照回 Engine，禁止 Worker 直接写 Engine 内存。

5. **教学顺序**  
   Local 单卡 eager → 单卡 CG → TP（层内 NCCL）→ PP（stage 张量）→ VL → MoE。

### 可演进但不推翻的点

- MoE fused 按 arch 自动降级，避免能力表与实现脱节；  
- decode RPC 继续瘦身，降低 TTFT；  
- ViT CG 作为可选加速，默认关闭以保持路径简单。

---

## 7. 和调度流并排对照

| | 调度流 | 计算流 |
|--|--------|--------|
| 进程 | Engine | Worker（或同进程 Local） |
| 核心对象 | `Sequence` 队列、`BlockManager` | `ModelRunner`、KV pool、weights |
| 一步输入 | 全局 waiting/running | 本步 `seqs` + `is_prefill` |
| 一步输出 | `SchedulerOutput` | `token_ids` (+ mrope) |
| 通信 | 无（或仅被 Engine 调用） | NCCL +（经桥）RPC |
| 禁止事项 | 调用 `model.forward` | 修改 waiting/running |

完整系统 = **调度流决策 → 桥投递 → 计算流执行 → 调度流提交**。两张图合在一起，就是 v7 的运行时灵魂。
