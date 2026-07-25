# vllm-v4 DESIGN — 在 v3 上的高性能多模态（Qwen3-VL）

> **定位：** `v4 = v3（CB + FlashInfer + Triton + Decode CG + hash/radix APC + PD mix） + 官方级多模态合同 + 最高效 encode/decode/算子路径`  
> **零件来源：** v5（ViT / Qwen3-VL / DeepStack / MRoPE 语义）、v7（按窗 encode、图前前缀、prepare-once）、官方 vLLM v1（Encoder 调度预算、EncoderCache、ViT attn wrappers、Encoder CG）  
> **不做：** TP/PP/EP、EPD 分实例、LLaVA/InternVL 全家桶、MoE+VL（留给后续版本）

本文是实现蓝图，不代替代码。实现时应 **fork/复制 v3 树再增量**，而不是从 v5 整树拷贝。

---

## 0. 目标与成功标准

### 0.1 功能成功

| # | 标准 |
| --- | --- |
| F1 | 单卡跑通 Qwen3-VL：`add_request(prompt, images=[...])` → 正确图文输出 |
| F2 | Chunked prefill 可跨 image span；DeepStack + interleaved MRoPE 与 HF 短序列对齐 |
| F3 | 文本模型路径零回归（Qwen3 纯文本仍走 v3） |
| F4 | hash **与** radix 均支持 **图前文本** APC；EncoderCache 按 `mm_hash` 跨请求共享 ViT 输出 |
| F5 | Preempt 后 `reset_mrope_state`；含图页永不进入 KV 前缀索引 |

### 0.2 性能成功（相对 v5，同卡同模型）

| # | 标准 |
| --- | --- |
| P1 | 纯文本 decode：不低于 v3（FI knobs + CG 保持） |
| P2 | 多图 / 高分辨率：ViT Attention **合批 varlen**，禁止 per-image SDPA 循环 |
| P3 | 重复图像：EncoderCache 命中后 **ViT≈0** |
| P4 | Chunked：仅对窗口重叠且未缓存的图 encode（按窗） |
| P5 | 可选 ViT CG（budget pack）+ 默认开启的 decode CG 路径 |
| P6 | PD mix 下 decode 不被「整步只做 prefill」饿死（继承 v3，优于 v5/v7） |

### 0.3 非目标（v4 明确砍掉）

- Vision DP / Weight-TP、多进程 PixelStore RPC、EPD 解耦  
- 图像页 content-addressed KV APC、投机解码、量化  
- 把 LM 与 ViT 塞进同一张 CUDA Graph  

---

## 1. 架构总览

```text
                         ┌─────────────────────────────────────────┐
                         │              vllm-v4 Engine               │
                         └─────────────────────────────────────────┘
                                            │
          ┌─────────────────────────────────┼─────────────────────────────────┐
          ▼                                 ▼                                 ▼
   ┌──────────────┐              ┌──────────────────┐              ┌──────────────────┐
   │ Scheduler    │              │ KVManager        │              │ ModelRunner      │
   │ (v3 + MM)    │              │ (v3 + MM cap)    │              │ (v3 LM + Vision) │
   │              │              │                  │              │                  │
   │ token budget │              │ pages + KV pool  │              │ decode: FI+CG    │
   │ PD mix       │              │ hash|radix APC   │              │ prefill: encode  │
   │ enc budget   │◄────────────►│ text-before-MM   │              │   → merge → LM   │
   │ enc schedule │              │                  │              │                  │
   └──────┬───────┘              └──────────────────┘              └────────┬─────────┘
          │                                                                  │
          │  scheduled_encoder_inputs                                         │
          ▼                                                                  ▼
   ┌──────────────┐                                              ┌──────────────────┐
   │ EncoderCache │◄──── mm_hash 共享 ───────────────────────────│ Vision stack     │
   │ Manager      │                                              │ ViT attn (FA/FI) │
   │ (refcount)   │                                              │ optional ViT CG  │
   └──────────────┘                                              │ MRoPE prepare×1  │
                                                                  └──────────────────┘
```

**三面合同不变（v3）：** 控制 / 资源 / 数据。  
**新增第四条轴：** Encoder（调度预算 + 内容缓存 + 视觉算子），与 Decoder 轴正交。

---

## 2. 目录与模块地图

以 **复制 v3** 为起点，增量如下。

### 2.1 原样保留（禁止为了 MM 而回退）

| 路径 | 原因 |
| --- | --- |
| `layers/attention*.py` | LM FlashInfer paged |
| `memory/kv_ops.py` | Triton store |
| `layers/layernorm.py`, `rotary.py`, `activation.py` | Triton / compile |
| `core/cuda_graph.py` | Decode CG |
| `engine/sampler.py`, `input_builder.py` 文本路径 | pin + batch |
| `config.py` 中 FI knobs、`mix_prefill_decode`、`prefix_backend` | 性能旋钮 |

### 2.2 新增包（已实现）

实际落地时把编排收敛进 `models/qwen3_vl.py`（塔 + LM + `encode_and_merge`），
避免多一层 `vision/` 运行时包；`EncoderCache` 与视觉塔并列在 `multimodal/vision.py`。
mm_hash 直接在 `engine._build_mm_features` 里对像素内容做 sha1（无需单独 hasher）。

```text
src/multimodal/                 # CPU + 视觉塔
  processor.py                  # HF AutoProcessor → ids + pixels + spans
  vit.py                        # Qwen3 ViT + merger + DeepStack（FI ragged / SDPA 双路）
  vision.py                     # Qwen3VisionTower + LRU EncoderCache（mm_hash→embeds+DeepStack）
  _mrope.py                     # LM 侧 3 轴 MRoPE 位置（prepare-once + rope_delta）

src/layers/vision/
  vit_attn.py                   # ★ FlashInfer ragged ViT attention 后端（见 §5）

src/layers/mrope.py             # interleaved MRoPE（LM 侧，复用 v3 Triton RoPE kernel）
src/models/qwen3_vl.py          # 塔 + LM + encode_and_merge 编排 + 权重加载
src/core/vit_cuda_graph.py      # 可选 budget CG（capture 期回退 SDPA）
```

> 说明：`MultiModalFeature` 直接放在 `memory/sequence.py`（与 `Sequence` 同源，
> 避免循环依赖）；`overlaps_window` / `truncate` 逻辑分别内联进
> `qwen3_vl.encode_and_merge` 的重叠窗口计算与 `engine._truncate_left`；
> encoder 预算/驱逐用 `EncoderCache` 的 LRU 上限承担（教学单卡足够，未做调度侧 refcount）。

### 2.3 小改扩展

| 文件 | 变更要点 |
| --- | --- |
| `memory/sequence.py` | MM 字段；`prefix_cacheable_full_blocks()`；`reset_mrope_state()` |
| `memory/prefix/hash_index.py` | match/publish 用 cacheable 长度 |
| `memory/prefix/radix_index.py` | **同一函数**（hash/radix 合同一致） |
| `memory/manager.py` | 池维度用 `text_config`；sync 尊重 cap |
| `engine/scheduler.py` | encoder 预算；`scheduled_encoder_inputs`；miss 缩短窗口；preempt reset |
| `engine/model_runner.py` | MM prefill：runner.encode → LM；decode 仍 FI/CG |
| `engine/engine.py` | `images=`；截断；文本/VL 模型加载 |
| `engine/input_builder.py` | decode `+ rope_delta` |
| `config.py` | MM / ViT CG / enc budget；保留全部 v3 旋钮 |
| `models/qwen3.py` | `inputs_embeds` + DeepStack 注入点 |

---

## 3. 控制面设计（学官方 Scheduler）

### 3.1 继承 v3 的优势

```text
默认 mix_prefill_decode=True
  ① 先尽量排 decode
  ② 剩余 token budget 排一块 prefill（可含 VL）
→ 比 v5/v7「整步 prefill-first」更利于 decode 延迟
```

### 3.2 增加 Encoder 调度（对齐官方 `_try_schedule_encoder_inputs`）

每步额外维护：

| 量 | 含义 |
| --- | --- |
| `encoder_compute_budget` | 本步最多新算多少 visual tokens |
| `encoder_cache` 空闲槽 | embeds 容量（按 token 或字节，推荐双限制） |
| `scheduled_encoder_inputs` | `{seq_id: [mm_index, ...]}` |

规则：

1. 仅当 MM span 与本步 LM 窗口 `[cached, cached+new)` **重叠**，且 cache miss，才进入 encode 列表。  
2. 整图 **原子** encode（ViT 双向）；默认不把一张图拆成多次 ViT forward。  
3. 若 budget/cache 不够 → **缩短** `num_scheduled_tokens`，停在该图之前（不得「先跑过未编码的 image token」）。  
4. Cache hit：不占 compute budget，只保证 cache pin/refcount。  
5. 完成/抢占：按官方语义 `free` 无引用 hash（可延迟驱逐）。

### 3.3 Schedule 输出（概念）

```text
ScheduleResult:
  prefills: list[Sequence]      # 与 v3 相同
  decodes:  list[Sequence]
  encoder_inputs: dict[seq_id, list[mm_idx]]
  # ModelRunner.run 顺序建议：
  #   1) (可选 P2) 异步：启动 encoder_inputs 的 ViT
  #   2) decode forward（FI / CG）
  #   3) 等待 encoder 完成（若与 decode 重叠）
  #   4) prefill forward（merge 已就绪的 embeds）
```

P0 可串行：`encode → decode → prefill`（仍优于「先全 prefill」）。  
P2 再做：`encode ∥ decode` 双流。

### 3.4 Preempt

与 v3 相同的 KV 回收，并强制：

```text
seq.reset_mrope_state()
# EncoderCache：按 mm_hash 可保留（跨请求复用）；不要绑死在 seq_id
```

---

## 4. 资源面：双缓存合同

### 4.1 KV APC（hash 与 radix 必须一致）

```text
first_mm = min(mf.start) or +∞
cacheable_full_blocks =
    严格位于 first_mm 之前的、已完成的满页数量

match / publish / pin：只使用 cacheable_full_blocks
含图及之后的页：私有，不进 HashIndex / 不进 RadixTree
```

实现：**唯一真相**放在 `Sequence.prefix_cacheable_full_blocks()`；两个 index 只读它。  
驱逐 / invalidate / radix leaf LRU / hash 页 LRU：**沿用 v3，不改算法。**

### 4.2 EncoderCache（第二缓存）

| 项 | 设计 |
| --- | --- |
| Key | `mm_hash`（像素内容，不是 placeholder token） |
| Value | merged visual embeds + DeepStack 特征 |
| 驻留 | 推荐 **CPU 大容量 + 使用时 H2D**（学 v7，省 GPU）；热点可 pinned |
| 上限 | `max_num_encoder_cache_tokens` + `max_encoder_cache_bytes` |
| 驱逐 | 无引用 + LRU（对齐官方 EncoderCacheManager 思想） |

与 KV APC **正交**：同一张图不同 system prompt → 共享 ViT，不共享图后 KV。

---

## 5. 算子层：Encode / Decode 都要「最高效」

### 5.1 Decoder（LM）— 锁死 v3 峰值，禁止回退

| 算子 | v4 要求 |
| --- | --- |
| Attention | FlashInfer paged；`use_tensor_cores=True`；`non_blocking=True` |
| KV write | Triton `store_kvcache` |
| RMSNorm / RoPE(text) / Silu | Triton / compile 保持 |
| Decode CG | 默认可开；VL decode 仅用 `rope_delta` 的 1D 位置进图 |
| H2D / Sampler | pin_memory + 整批采样 |
| Prefill LM | FlashInfer prefill + varlen `cu_seqlens`（已有） |

**硬约束：** 加载 VL 模型不得把 FI knobs 改成 False（v5 曾硬编码 `use_tensor_cores=False`，v4 **禁止**）。

### 5.2 Encoder（ViT Attention）— 对齐官方 `vit_attn_wrappers`

现状（v5/v7）：`for image: SDPA` → 必须消灭。

**v4 默认后端优先级：**

```text
1) flash-attn varlen（causal=False）     # 官方主路径，成熟
2) FlashInfer ragged / vit wrapper       # 与 LM 栈统一时可选
3) 打包 SDPA / flex 仅作 fallback
```

接口形状（概念）：

```text
q,k,v: packed [sum_tokens, H, D]
cu_seqlens: int32[B+1]   # 每张图一段
max_seqlen: int
→ out: [sum_tokens, H, D]
```

ViT 内已有 `cu_seqlens` / seqlens / offsets → 直接喂合批核。  
RoPE：保持「先旋再 attn」；attn 核不负责 MRoPE。

### 5.3 其它视觉算子

| 模块 | 优化 |
| --- | --- |
| Patch embed / MLP | 保持融合 QKV 权重加载（v5/v7 已有） |
| Bilinear pos embed | Triton（已有）优先 |
| Merger / DeepStack proj | 标准 Linear；避免多余 clone（merge 用 in-place scatter） |
| 多图 miss | `torch.cat` 一次 ViT forward（已有）+ **合批 attn** |

### 5.4 Encoder CUDA Graph（可选，默认关或按常见分辨率开）

对齐官方 budget CG：

```text
budgets = power-of-2 visual token 容量（含模型 max）
runtime: 按图 token 数从小到大 greedy pack → 选最小能装下的 budget → replay
装不下 → eager
```

与 Decode CG **正交**：可同时启用。  
Capture 期间禁止增长 MRoPE cos/sin 表。

### 5.5 MRoPE

- Prefill：`prepare(positions_3d)` **每步一次**，各层只 `apply`（学 v7，禁止每层重建表）  
- Decode：只用 `rope_delta`，进 CG  

### 5.6 流水线重叠（性能上限，P2）

```text
stream_lm:     decode batch（FI / CG）
stream_vit:    本步 scheduled_encoder_inputs 的 ViT
event:         VitDone
然后:          prefill LM 消费 embeds
```

同请求内 `ViT → LM prefill` **不可**自重叠。  
语言-only 请求 **零** encoder 开销。

---

## 6. 端到端数据流

### 6.1 文本请求

与 v3 完全相同（前缀 / PD mix / FI / CG）。

### 6.2 VL 请求

```text
Engine.add_request(prompt, images)
  → Processor → token_ids + mm_features(mm_hash, span, cpu pixels)
  → Sequence

step:
  schedule → (decodes, prefills, encoder_inputs)
  allocate KV：prefix match 仅图前文本

  [P0] encode(encoder_inputs) 串行
  [P2] encode ∥ decode

  decode forward（若有）
  for prefill seqs:
      merge embeds from EncoderCache into window
      build DeepStack window buffers
      MRoPE positions + state update
      LM forward(inputs_embeds, pos[3,T], deepstack)
      sample if prompt finished else skip LM-head（v3 已有）

  postprocess → sync_prefix（只 publish cacheable 页）
```

### 6.3 Chunked 跨图

1. 第一次窗口碰到图 → 整图 encode（或 cache hit）→ 设 `mrope_origin`  
2. 后续 chunk → 只 **切片** cache 中的 embeds  
3. 图结束后文本位置从 `max(pos)+1` 延续（HF `rope_deltas` 语义）

---

## 7. 与 v5 / v7 / 官方对照（设计选型）

| 议题 | v5 | v7 | 官方 | **v4 选型** |
| --- | --- | --- | --- | --- |
| 底座 | 偏旧文本栈 | 重并行 | v1 | **v3 全文底座** |
| PD | prefill-first | prefill-first | 混合 token | **v3 PD mix** |
| 前缀 MM | 全关 | 图前文本 | 图前 + EC | **图前 + EC（hash+radix）** |
| Encode 调度 | Runner 内 | 按窗 | 预算+缩短窗口 | **官方合同精简版** |
| ViT attn | SDPA 循环 | SDPA 循环 | FA/FI wrappers | **FA/FI 合批** |
| ViT CG | 有，窄 | 有，默认关 | budget pack | **可选 budget CG** |
| FI TC | 常 False | True | True | **强制可配且默认 True** |
| MRoPE | 层内易重复 | prepare-once | — | **prepare-once** |
| EPD | 无 | 无 | 有 | **v4 不做** |
| 编排位置 | 进 model | vision/ | worker mm/ | **vision/encoder_runner** |

---

## 8. 配置面（建议）

```text
# 继承 v3
prefix_backend: none|hash|radix = hash
mix_prefill_decode: bool = True
flashinfer_use_tensor_cores: bool = True
flashinfer_non_blocking: bool = True
enforce_eager: bool = True          # False → decode CG

# v4 新增
max_num_encoder_input_tokens: int   # 每步 encode compute budget
encoder_cache_size: int             # embed token 容量
encoder_cache_bytes: int            # 字节上限（0=不限制）
vit_attn_backend: auto|flash|flashinfer|sdpa = auto
enable_vit_cuda_graph: bool = False
vit_cg_token_budgets: list[int] | auto
enable_encode_decode_overlap: bool = False   # P2
```

---

## 9. 实现阶段（必须按序，保证「最高效」可交付）

### Phase 0 — 树与合同（0.5–1 天量级）

1. 复制 `vllm-v3` → `vllm-v4`  
2. 落地本文 DESIGN；README 写清「v3+MM，不是 v5 改名」  
3. 文本回归测试先绿  

### Phase 1 — 正确完整 MM（P0）

1. multimodal CPU + Sequence + qwen3_vl + MRoPE + DeepStack  
2. EncoderCache + **按窗** encode + runner 编排  
3. Scheduler：encoder_inputs + miss 缩短窗口（可先粗预算）  
4. hash **与** radix 的 `prefix_cacheable_full_blocks`  
5. preempt reset；HF 对齐；chunked 跨图；文本零回归  

**此时允许 ViT 暂用 SDPA，但接口必须预留 `vit_attn.py`。**

### Phase 2 — 算子打满（性能主峰）

1. `vit_attn`：flash-attn varlen / FI，去掉 for-SDPA  
2. 确认 LM FI knobs 与 decode CG 在 VL 下不回退  
3. MRoPE prepare-once；merge in-place；EncoderCache 字节帽  
4. Bench：多图 TTFT、重复图 cache hit、文本 decode vs v3  

### Phase 3 — Encoder CG + 重叠（上限）

1. Budget ViT CG  
2. `enable_encode_decode_overlap` 双流  
3. 文档写清命中条件与回退  

---

## 10. 测试与 Bench 清单

### 正确性

- `verify_qwen3_vl_hf`（ViT / DeepStack / MRoPE / 短 greedy）  
- 多图、chunked 跨图、preempt 再跑  
- 图前 APC：hash 与 radix 各测 shared system + 不同图  
- 纯文本 Qwen3：v3 指标回归  

### 性能

| Bench | 看什么 |
| --- | --- |
| 文本 decode CG | ≥ v3 |
| 单图 eager vs FA ViT | ViT 段加速 |
| 多图合批 | 相对 v5 for-SDPA |
| 同图二次请求 | encode≈0 |
| PD mix：decode+VL prefill | decode 不被饿死 |
| （P3）overlap on/off | 利用率 |

---

## 11. 风险与不变量（实现时贴在 PR 模板）

1. **未 encode 不得越过 image token。**  
2. **含图页不得 publish/match 进 APC。**  
3. **Preempt 必须 reset MRoPE。**  
4. **KV 池用 text_config 层数/头。**  
5. **Decode 不跑 ViT / 不加 DeepStack。**  
6. **VL 不得关闭 FI Tensor Core / 破坏 decode CG 契约。**  
7. **ViT CG capture 中不扩 MRoPE 表。**  

---

## 12. 一句话交付物

> **vllm-v4 = 以 v3 为唯一性能底座的官方风格多模态教学实现：Encoder 调度+双缓存（KV 图前 APC + mm_hash EC）+ ViT 合批算子/可选 CG + LM FlashInfer/Decode CG/PD mix 全保留；零件取自 v5/v7，合同取自官方 vLLM v1。**

实现顺序：**先合同与正确性（Phase 1）→ 再 ViT/算子打满（Phase 2）→ 最后 CG/双流（Phase 3）**。  
任何「先抄 v5 能跑」但关掉前缀或 FI knobs 的路径，都视为 **设计不合格**。
