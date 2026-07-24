# vllm-v3 优化全景与继续优化指南（保留 FlashInfer）

本文回答三件事：

1. **v3 现在已经做了哪些优化**  
2. **在你当前性能画像下，下一步该怎么排优先级**  
3. **坚持 FlashInfer 时，CUDA Graph replay 该怎么改**（不要切到 flash-attn）

---

## 1. v3 优化全景

v3 = **v2 Continuous Batching 控制面** + **高性能数据面**。

```
┌─────────────────────────────────────────────────────────┐
│ Engine.step / Scheduler / BlockManager / Sequence       │  ← 控制面（CB + PD mix）
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│ ModelRunner 数据面（v3 主战场）                            │
│  pin_memory 输入 → FlashInfer Attn → Triton KV/Norm/RoPE │
│  → batched Sampler → (可选) Decode CUDA Graph             │
└─────────────────────────────────────────────────────────┘
```

### 1.1 已落地清单

| # | 优化 | 位置 | 解决什么问题 | 主要收益场景 |
|---|------|------|--------------|--------------|
| A | FlashInfer paged prefill/decode | `layers/attention*.py` | 替换 PyTorch gather+matmul | 中长 prefill、批 decode |
| B | Triton `store_kvcache` | `memory/kv_ops.py` | 替换 `index_copy_` scatter | 每步 KV 写入 |
| C | Triton fused RMSNorm（含 residual） | `layers/layernorm.py` | 替换纯 PyTorch norm | 每层 pre/post norm |
| D | Triton RoPE | `layers/rotary.py` | 替换纯 PyTorch rotate | 每层 Q/K |
| E | `torch.compile` SiluAndMul | `layers/activation.py` | 融合 silu×mul | MLP |
| F | `block_size=256` | `config.py` | FlashInfer/页对齐约定 | 启用高性能后端前提 |
| G | 动态 KV 池 | `model_runner._alloc_kv` | 按显存利用率 sizing | 显存利用率 / 并发容量 |
| H | pin_memory + non_blocking H2D | `input_builder.py` | 输入拷贝与计算重叠 | 组批开销 |
| I | 批量 Sampler | `sampler.py` | 多请求一次采样 | batch decode |
| J | Decode CUDA Graph | `core/cuda_graph.py` | 消除 decode kernel launch | **小模型单请求/小 batch decode** |
| K | Prefill varlen batch | FlashInfer prefill + cu_seqlens | 真正按 token 拼 batch | continuous batching |

### 1.2 相对 v2 / nano 的实测画像（你机器上）

**vs v2（Qwen3-1.7B）**

| 指标 | 结论 |
|------|------|
| TTFT | 接近或略优（短长 prompt 差不大） |
| Decode / E2E | **~1.7×**（算子换血生效） |
| Batch n=8 | **~3.3×** |
| +CG | decode 再跳到数百 tok/s |

**vs nano（Qwen3-0.6B quick）**

| 路径 | 结论 |
|------|------|
| eager | v3 **略快**（~1.3×） |
| +CUDA Graph | nanocg **略快**（~1.3×）——差距在 **CG replay 的 FlashInfer plan 税**，不在「会不会 capture」 |

---

## 2. Decode 为什么开 CG 会「突然」起飞？

小模型一步 decode ≈ **几十个小 kernel**：

```
embed + N×(norm, qkv, rope, attn, o, norm, mlp) + lm_head
```

eager 时：GPU 算得飞快，大量时间在 **等 CPU 下发下一个 kernel**（launch-bound）。

CUDA Graph：把整段 forward **录成一张图**，每步几乎只是：

1. 写固定 buffer  
2. `graph.replay()`  

从 launch-bound → compute-bound，所以 30–60 tok/s → 300–400+ tok/s，看起来「突变」，本质是 **去掉启动税**。

Prefill（TTFT）通常不明显，因为矩阵够大，本来就不那么吃 launch。

---

## 3. 坚持 FlashInfer 时，CG 慢在哪？

nano 的 CG：

```
拷贝 input/slot/block_table → replay
```

attention 用 `flash_attn_with_kvcache`，**元数据本身就是图里的静态 tensor**，无需每步「重新规划」。

v3 / FlashInfer decode：

```
拷贝 input → plan(indptr, indices, last_page_len) → replay
```

`BatchDecodeWithPagedKVCacheWrapper.plan()` **每步必做**，且库内部还要把 `indptr` / `last_page_len` 弄到 **host** 做调度表。这是 FlashInfer paged decode + CG 的固有形态，换 flash-attn 能绕开，但你要留 FlashInfer，就只能 **把 plan 税压到最低**，而不能指望「完全不 plan」。

你原来的 `_replan_cg` 额外踩了坑：

| 坑 | 影响 |
|----|------|
| `indptr[real_bs].item()` | **每步强制 GPU sync** |
| `indptr.cpu()` / `last_page.cpu()` | 多余 D2H + 临时张量 |
| 每次 `torch.zeros` / `cat` / 新建 metadata | Python + 分配开销 |
| `padded.zero_()` 整段 indices | 多余清零 |

→ Graph 很快，**图外的 plan 把 CG 红利吃掉一截**，所以 v3cg < nanocg。

---

## 4. FlashInfer CG replay：推荐优化路线（已部分落地）

### P0 — 立刻做（高收益 / 仍用 FlashInfer）✅ 已落地

文件：`src/core/cuda_graph.py`

1. **Pinned host buffer** 固定存放 `indptr` / `last_page_len` / `indices`
2. **零 GPU sync**：`replay(..., seq_lens=, block_tables=)` 直接吃 CPU `Sequence` 状态
3. 禁止 `.item()` / `.cpu()` / 从 GPU 反查页表
4. `plan(..., non_blocking=True)` + H2D indices
5. 复用 per-bs `AttentionMetadata`
6. BS 阶梯与 nano 对齐：`[1,2,4,8] + 16…`
7. 去掉无用的 GPU `context_lens` / `block_tables` 每步拷贝（FlashInfer 页表只走 plan）

### P1 — 继续压 plan 税（推荐优先）

| 改动 | 做法 | 预期 |
|------|------|------|
| **零 sync replan** | `replay` 额外接收 CPU `seq_lens: list[int]`、`block_tables_cpu: list[list[int]]`（Scheduler/Sequence 本来就有），完全不读 GPU `context_lens` | 再砍 1 次 D2H |
| **增量 last_page** | 同 batch 连续 decode：多数步只是 last_page +=1 或换新 page；可只改 dirty 项 | 小 bs 减 CPU 循环 |
| **CPU 填 indices** | 小 `max_num_seqs` 时在 host 拼 indices，一次 H2D；避免 GPU mask gather | 小 batch 更稳 |
| **减少 pad 阶梯** | bench/服务若 `max_num_seqs≤8`，CG 只 capture `[1,2,4,8]`，少占显存、少换图 | 启动与显存 |

### P2 — FlashInfer 后端旋钮

| 旋钮 | 说明 |
|------|------|
| `use_tensor_cores=True` | 大 head / 较大 bs 时可能更快；小模型 bs=1 要实测 |
| Prefill workspace | 已共享 256MB workspace；可按模型尺寸调 |
| 固定 `num_kvcache_blocks` | 生产上避免动态估太大拖慢 alloc；bench 已支持覆盖 |

### P3 — 层算子（decode bs=1 路径）

FlashInfer 决定大头后，层算子在 **CG 内**会被一起 capture，差异变小；eager 路径仍有意义：

| 算子 | 建议 |
|------|------|
| RMSNorm / RoPE | 对 **bs=1 eager** 对比 Triton vs `@torch.compile`（nano 风格）；CG 内两者都进图，差异更小 |
| SiluAndMul | 已 compile，保留 |
| Linear | 教学向可暂不引入 fused GEMM / FP8 |

### P4 — 调度与输入（与 nano 打平「系统」）

| 项 | 现状 | 可选 |
|----|------|------|
| Prefill-first CB | ✅ | 保持 |
| 哈希前缀 | ✅ | 多请求重复前缀再测命中 |
| 输入组 batch | pin_memory ✅ | decode 可减少 Python list 拼装 |
| Torch 整模 compile | 无 | 一般 **不要** 和 CG 叠太多；nano 也是层内 compile |

---

## 5. 你现在应该怎么排优先级？

按 **你要追的指标** 选：

### 目标 A：追 nanocg 的 decode 吞吐（仍用 FlashInfer）

```
1. 测当前 P0 replan 优化后的 v3cg（本仓库）
2. P1：CPU lens/block_table 进 replay，彻底去掉 GPU sync
3. P1：缩小 CG bs 阶梯
4. 再 profile：nsys 看 plan 是否仍占 decode 步长 >20%
```

**不要**为了 CG 去换 flash-attn——差距可用 FlashInfer plan 路径工程化收掉大部分。

### 目标 B：服务吞吐（多请求 continuous batching）

```
1. 保持 v3 eager 或 v3cg（batch↑ 时 CG 仍有效）
2. 调大 max_num_seqs / max_num_batched_tokens
3. 动态 KV 池 + 前缀缓存命中率
4. Prefill FlashInfer varlen 已经是对的方向
```

这类场景 v3 相对 v2 已经很强；相对 nano 通常也接近。

### 目标 C：TTFT / 长 prompt

```
1. FlashInfer prefill 已基本对
2. TTFT 差通常不在 CG；看 H2D / tokenizer / 调度
3. 长 context 再对照 FlashInfer causal prefill 配置
```

---

## 6. FlashInfer CG replay：理想伪代码

```text
# capture（一次）
wrapper = BatchDecode(..., use_cuda_graph=True,
                      indptr_buf=GPU, indices_buf=GPU, last_page_buf=GPU)
wrapper.plan(初始 host 元数据)
with torch.cuda.graph(g):
    model(static_inputs)   # 内含 wrapper.run

# replay（每步）—— 目标形态
host_indptr[:], host_last[:] = from_cpu(seqs)   # 无 GPU sync
indices_buf[:npages] = ...                      # 或 CPU 拼好 H2D
wrapper.plan(host_indptr, indices_buf[:npages], host_last, non_blocking=True)
copy static input_ids / positions / slot_mapping
g.replay()
```

关键约束（FlashInfer 决定，甩不掉）：

- **每步仍要 `plan`**（更新 host 调度 + 把页表拷进 CG buffer）  
- CG 只能固定 **batch 阶梯**（所以要 pad）  
- Prefill shape 变 → **不要**对 prefill 做 CG  

---

## 7. v3 的产品定位（避免优化跑偏）

| v3 是 | v3 不是 |
|-------|---------|
| 教学序列里「算子优化章」 | 完整生产 vLLM |
| FlashInfer + Triton + CG 的可讲实现 | 必须字节级打赢 nano |
| 在正确 Scheduler 上验证算子收益 | 追求所有微基准第一 |

优化原则：**先砍 CG 图外开销（plan/sync）→ 再调 FlashInfer 旋钮 → 最后动层算子**。  
整模大改或换 flash-attn 都不符合「继续用 FlashInfer」的目标。

---

## 8. 建议验证命令

```bash
# CG decode / batch
source dzyy_vllm/.venv/bin/activate
python scripts/compare_v3_nano_bench.py --model Qwen3-0.6B \
  --suites decode e2e batch --with-cuda-graph --iters 3

# 同模型 vs v2（确认没回退）
python scripts/compare_v2_v3_bench.py --model Qwen3-0.6B \
  --suites decode batch --with-cuda-graph --quick
```

---

## 9. 相关代码入口

| 主题 | 文件 |
|------|------|
| FlashInfer backend | `src/layers/attention_backend.py` |
| Attn + KV write | `src/layers/attention.py`, `src/memory/kv_ops.py` |
| CG + replan | `src/core/cuda_graph.py` |
| 调度 | `src/engine/scheduler.py` / `docs/scheduling.md` |
| 算子概览 | `docs/operators.md` |
