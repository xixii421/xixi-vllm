# vllm-v4 架构详解：从文本引擎到多模态引擎

> **一句话定位：** `v4 = v3（FlashInfer + Triton + Decode CG + hash/radix + PD mix）+ Qwen3-VL 多模态`  
> **核心思想：** 在 v3 的高性能文本底座上增量添加多模态能力，文本路径零回归，多模态路径走最高效的编码/解码/算子路径。

本文让你对 v4 的整体架构、模块依赖、多模态实现机理、以及与 v3 的差异有一个系统性的理解。

## 1. 版本演进与代码量

### 1.1 版本线

| 版本 | 主题 | src/ 行数 | Python 文件数 |
|------|------|-----------|--------------|
| v0 | 最简文本推理循环 | ~738 | 18 |
| v1 | KV Cache + 页表 | ~1,887 | 28 |
| v2 | Continuous Batching + 前缀缓存 | ~2,106 | 32 |
| v3 | FlashInfer 算子 + Triton + Decode CG | ~3,436 | 36 |
| **v4** | **多模态 Qwen3-VL** | **~12,558** | **75** |

从 v3 到 v4，代码量增长了约 **3.6 倍**，增量几乎全部来自多模态相关模块（`src/multimodal/`、`src/layers/vision/`、`src/models/qwen3_vl.py` 等）以及大量的基准测试脚本。

### 1.2 v4 各模块代码量分布

| 模块 | 行数 | 占比 | 说明 |
|------|------|------|------|
| `bench/` | ~5,000+ | ~40% | 性能基准测试（多版本对比、profile、CG suite） |
| `src/multimodal/` | ~1,355 | ~11% | 多模态核心：processor + vit + vision + mrope |
| `src/engine/` | ~1,323 | ~11% | 调度器、模型执行器、输入构造器 |
| `src/memory/` | ~1,031 | ~8% | KV 管理器、前缀缓存（hash/radix） |
| `src/models/` | ~773 | ~6% | 模型定义：qwen3 + qwen3_vl + 权重加载 |
| `src/layers/` | ~458 | ~4% | 算子层：attention、RoPE/MRoPE、RMSNorm、activation |
| `tests/` | ~1,650 | ~13% | 测试套件 |
| `src/core/` | ~299 | ~2% | CUDA Graph 执行器 |

---

## 2. 架构总览

v4 延续 v3 的 **三面合同**（控制面、资源面、数据面），并增加第四条轴：**Encoder 面**。

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

**四条轴：**

1. **控制面（Scheduler）**：在 v3 的 `mix_prefill_decode` 基础上，新增 encoder 预算管理。调度时如果图片未编码且预算不足，会缩短 prefill chunk 停在图片之前。
2. **资源面（KVManager + EncoderCache）**：KV 前缀缓存 + 视觉编码缓存，两套独立的缓存系统。KV APC 只缓存"第一张图之前"的文本前缀；EncoderCache 按像素内容 hash（`mm_hash`）跨请求复用 ViT 输出。
3. **数据面（ModelRunner）**：decode 仍走 v3 的 FlashInfer + Decode CG；prefill 在遇见多模态请求时走 `encode_and_merge`（ViT 编码 → scatter embeds → MRoPE → LM forward）。
4. **Encoder 面（Vision Tower + EncoderCache）**：新增维度，负责视觉编码的调度预算分配、内容缓存、合批 inference 和可选的 CUDA Graph 加速。

---

## 3. 模块地图与依赖关系

### 3.1 所有模块一览

```text
src/
├── config.py                         # EngineConfig：VL 检测 + 性能旋钮
├── sampling_params.py                # 采样参数
│
├── core/
│   └── cuda_graph.py                 # Decode CUDA Graph 捕获/回放（v3 继承）
│
├── engine/
│   ├── engine.py                     # 主引擎：请求管理、截断、模型加载
│   ├── scheduler.py                  # 调度器：PD mix + encoder 预算
│   ├── model_runner.py               # 模型执行器：编解码编排
│   ├── model_inputs.py               # 模型输入数据结构
│   ├── input_builder.py              # 输入构造：batch tensors + rope_delta
│   └── sampler.py                    # 采样器：FlashInfer top-k/top-p
│
├── models/
│   ├── qwen3.py                      # 纯文本 Qwen3（v3 继承，增加 DeepStack 注入点）
│   ├── qwen3_vl.py                   # ★ Qwen3-VL 编排：塔+LM+encode_and_merge
│   ├── loader.py                     # 权重加载工具
│   └── hub.py                        # HuggingFace 模型下载
│
├── layers/
│   ├── attention.py                  # FlashInfer paged attention 包装
│   ├── attention_backend.py          # FlashInfer 配置和元数据构建
│   ├── activation.py                 # SiLU-and-Mul 融合算子
│   ├── layernorm.py                  # Triton RMSNorm
│   ├── rotary.py                     # Triton RoPE（1-D 文本位置）
│   ├── mrope.py                      # ★ 交错 MRoPE（3-D，复用 Triton RoPE kernel）
│   └── vision/
│       └── vit_attn.py               # ★ FlashInfer/FlashAttention ViT 注意力后端
│
├── memory/
│   ├── sequence.py                   # Sequence 状态 + MultiModalFeature + MM 字段
│   ├── block_manager.py              # KV 块分配器
│   ├── kv_cache.py                   # GPU 页表 + H2D 同步
│   ├── kv_ops.py                     # Triton K/V store 核
│   ├── manager.py                    # KVManager：池分配 + 前缀同步
│   └── prefix/                       # 前缀缓存后端
│       ├── hash_index.py             #   哈希索引
│       ├── radix_index.py            #   基数树索引
│       ├── radix_tree.py             #   基数树实现
│       ├── hash_util.py              #   xxhash 工具
│       └── noop.py                   #   空后端
│
├── multimodal/
│   ├── processor.py                  # HF AutoProcessor：token_ids + pixels + spans
│   ├── vit.py                        # ★ Qwen3 ViT 模型（patch embed、attention blocks、merger）
│   ├── vision.py                     # ★ 视觉塔 + EncoderCache + ViT CUDA Graph
│   ├── _mrope.py                     # ★ LM 侧 3 轴 MRoPE 位置计算（prepare-once）
│   ├── image.py                      # PIL 图像缩放工具
│   └── hasher.py                     # SHA-256 图像哈希
│
└── utils/
    └── context.py                    # 线程局部 forward 上下文
```

### 3.2 模块依赖图

```text
                         ┌──────────────┐
                         │   config.py  │ ← 所有模块的配置入口
                         └──────┬───────┘
                                │
        ┌───────────────────────┼───────────────────────────┐
        ▼                       ▼                           ▼
┌───────────────┐     ┌─────────────────┐       ┌──────────────────┐
│ engine.py     │────▶│ scheduler.py    │       │ model_runner.py  │
│ (主入口)       │     │ (调度决策)       │       │ (执行编排)         │
└───────┬───────┘     └────────┬────────┘       └────────┬─────────┘
        │                      │                         │
        │              ┌───────▼────────┐        ┌───────▼─────────┐
        │              │ memory/        │◄───────│ layers/         │
        │              │ KVManager      │        │ attention       │
        │              │ block_manager  │        │ rotary/mrope    │
        │              │ prefix/*       │        │ layernorm       │
        │              └───────┬────────┘        │ activation      │
        │                      │                 └─────────────────┘
        │                      │
        ▼                      ▼
┌───────────────────────────────────────────────────────┐
│                   models/qwen3_vl.py                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐│
│  │ Vision Tower │  │  Qwen3VLModel│  │ encode_and_  ││
│  │ + EncoderCache│  │  (LM layers) │  │   merge()   ││
│  └──────┬───────┘  └──────────────┘  └──────┬───────┘│
│         │                                    │        │
└─────────┼────────────────────────────────────┼────────┘
          │                                    │
          ▼                                    ▼
┌──────────────────┐                ┌──────────────────┐
│ multimodal/vit.py│                │ multimodal/      │
│ ViT 模型实现      │                │ _mrope.py        │
│ multimodal/      │                │ MRoPE 位置计算    │
│ vision.py        │                └──────────────────┘
│ 视觉塔 + ViT CG  │
└──────────────────┘
```

### 3.3 文本请求走哪些模块（v3 路径）

纯文本请求的完整链路：

1. `engine.py`：`add_request(prompt)` → `Sequence` 入队
2. `scheduler.py`：`schedule()` → `(prefills, decodes)`，分配 KV blocks
3. `model_runner.py`：`run(prefills, decodes)` →
   - decode：CUDA Graph replay 或 FlashInfer eager
   - prefill：FlashInfer prefill + varlen
4. `layers/`：`attention.py`（FlashInfer paged）、`rotary.py`（Triton RoPE）、`layernorm.py`（Triton RMSNorm）、`activation.py`（SiLU）、`kv_ops.py`（Triton KV store）
5. `sampler.py`：FlashInfer 采样
6. `memory/`：`prefix/`（hash/radix APC）、`block_manager.py`

**纯文本请求完全不碰多模态模块。**

### 3.4 多模态请求走哪些模块（新增路径）

带图片的请求链路：

1. `engine.py`：`add_request(prompt, images=[...])` →
   - `MultiModalProcessor.process(text, images)` →
     - `processor.py`：HF `AutoProcessor` → token_ids + pixel_values + spans
     - `hasher.py`：SHA-256 图像哈希 → `mm_hash`
   - `_build_mm_features()` → `MultiModalFeature`（挂载到 Sequence）

2. `scheduler.py`：`schedule()` → `_clamp_chunk_for_encoder_budget()`（encoder 预算检查）

3. `model_runner.py`：detect `has_multimodal` →
   - `model.encode_and_merge(input_ids, seqs, device)`：
     - `_encode_visual()`：查 EncoderCache（hit 直接复用）→ miss 则 batch encode
       - `vision.py`：`Qwen3VisionTower.encode()` → 可能走 ViT CUDA Graph（`VisionEncoderCudaGraph.try_encode()`）或 eager
       - `vit.py`：`Qwen3VisionTransformer.forward()` → patch embed → attention blocks（`vit_attn.py` FlashAttention varlen）→ merger → DeepStack 收集
     - 将视觉 embedding scatter 到 `inputs_embeds` 中
     - `_build_deepstack()`：构造 DeepStack 缓冲区
     - `_mrope.py`：`mrope_positions()` → 3 轴 MRoPE 位置
   - LM forward：`qwen3_vl.py` → `Qwen3VLModel.forward(inputs_embeds, deepstack_input_embeds, positions)`
     - `layers/mrope.py`：`MRotaryEmbedding.forward(positions_3d, q, k)` → 3 轴 RoPE
     - `layers/attention.py`：FlashInfer paged attention（与文本完全一致）
     - DeepStack 注入：前 N 层在 `hidden_states` 上加 `deepstack_input_embeds[i]`

---

## 4. 多模态核心实现详解

### 4.1 数据流全景：从图片到生成 token

```
用户传入 (prompt, images)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. MultiModalProcessor.process(text, images)                │
│    · PIL image → RGB + EXIF normalize                       │
│    · SHA-256 hash → mm_hash                                 │
│    · HF image_processor → pixel_values + image_grid_thw     │
│    · chat template → prompt → expand <|image_pad|> → token_ids │
│    · 解析 vision_start/end → MMPlaceholder(offset, length)  │
│    → MultiModalInputs(token_ids, pixel_values, placeholders) │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Engine._build_mm_features()                              │
│    · 为每张图构造 MultiModalFeature:                         │
│      - modality: "image"                                    │
│      - data: pixel_values (CPU pinned)                      │
│      - start/end: token_ids 中的视觉占位符位置               │
│      - num_encoder_tokens: 视觉 token 数量                   │
│      - uid: mm_hash                                         │
│      - grid_thw_tuple: (t, h, w) 三维网格信息                │
│    → Sequence.mm_features = [MultiModalFeature, ...]        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Scheduler._clamp_chunk_for_encoder_budget()              │
│    · 检查 prefill chunk 跨越的图片                            │
│    · 若 encoder budget 不足 → 在图片前截断                     │
│    · 保证："未编码不越过 image token"                           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. ModelRunner.run() → 检测 has_multimodal                  │
│    → Qwen3VLForCausalLM.encode_and_merge()                   │
│                                                             │
│    4a. _encode_visual(seqs)                                 │
│        · 遍历所有 mm_features，查 EncoderCache(mmf.uid)       │
│        · Cache HIT → 直接取 embeds（跳过 ViT）                 │
│        · Cache MISS → 收集所有 miss 的 pixel_values           │
│          → torch.cat → pin_memory → H2D                    │
│          → vision_tower.encode(pixel_values, grid_thw)       │
│             · [CG path] VisionEncoderCudaGraph.try_encode()  │
│             · [Eager path] Qwen3VisionTransformer.forward()  │
│               - patch_embed (Conv3d)                        │
│               - pos_embed (bilinear interpolation)           │
│               - attention blocks × N                        │
│                 · ViTAttentionBackend (flash_attn_varlen)    │
│                 · RoPE + QKV + attention + proj              │
│                 · 在 deepstack_indexes 层收集 DeepStack       │
│               - merger → 输出 visual embeddings              │
│          → VisionEncodeOutput(embeds, deepstack, sizes)     │
│        · 将编码结果写入 EncoderCache                            │
│                                                             │
│    4b. scatter: 将视觉 embeds 覆盖到 inputs_embeds 对应位置    │
│                                                             │
│    4c. _build_deepstack(): 构造 [ndps][N, H] 的 DeepStack 张量 │
│                                                             │
│    4d. mrope_positions(): 生成 [3, total_tokens] 的 3D 位置    │
│        · text 位置：base + arange（线性递增）                   │
│        · image 位置：替换为 (t, h/ms, w/ms) 的 grid 坐标       │
│        · image 结束后：位置从 max(visual_pos) + 1 继续         │
│        · 记录 rope_delta 和 mrope_next（chunked 续接用）       │
│                                                             │
│    → (inputs_embeds, deepstack_buffers, positions_3d)        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. LM Forward (Qwen3VLModel)                                │
│    · 每层：input_layernorm → attention → post_layernorm → MLP │
│    · attention: MRotaryEmbedding(positions_3d, q, k)        │
│      → 3 轴 interleaved MRoPE                               │
│      → FlashInfer paged attention（与文本相同）               │
│    · DeepStack: 在第 i 层（i < ndps）加 deepstack[i] 到 h     │
│    → final_norm(h) → hidden_states                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. lm_head → logits → Sampler → next_token                  │
│    · (与 v3 文本完全一致)                                      │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 MultiModalProcessor（图像预处理）

**位置：** `src/multimodal/processor.py`

这是多模态请求的入口，负责将图片 + 文本转换为模型可用的 token_ids、pixel_values 和 span 信息。核心流程：

1. **图像预处理**：`normalize_image`（EXIF 方向修正）+ `convert_image_mode(image, "RGB")`（RGBA→白底）
2. **内容哈希**：`hash_pil_image(image)` → SHA-256，作为 EncoderCache 的 key
3. **HF processor**：调用 `AutoProcessor.image_processor(images)` 得到 `pixel_values` 和 `image_grid_thw`
4. **Chat template**：构建 `[{"type": "image"}, {"type": "text"}]` 消息，`apply_chat_template` 生成带占位符的 prompt
5. **扩张占位符**：将单个 `<|image_pad|>` 替换为 `N` 个（N=编码后的视觉 token 数），然后 tokenize
6. **解析 span**：通过 `vision_start_token_id` 和 `vision_end_token_id` 定位每张图在 token_ids 中的起止位置

**设计要点：**
- 像素缓存：对相同图片，图片预处理结果（pixel_values + grid_thw）按 `(mm_hash, config_fingerprint)` 缓存，避免重复走 HF processor
- 配置指纹：`_fingerprint_config()` 对 image_processor 的关键参数取 SHA-1，确保配置变化时正确失效

### 4.3 EncoderCache（视觉编码缓存）

**位置：** `src/multimodal/vision.py`

这是 v4 实现"相同图片零编码"的关键模块。设计上对标本 vLLM 官方的 `EncoderCacheManager`：

**核心特性：**

| 特性 | 实现 |
|------|------|
| **缓存 key** | `mm_hash`（像素内容的 SHA-256，不是占位符 token） |
| **缓存 value** | visual embeds + DeepStack features |
| **容量管理** | 按 **token 数量**（不是 entry 数量）做预算 |
| **引用计数** | `retain(key, request_id)` / `release(request_id)` |
| **驱逐策略** | 优先驱逐零引用 entry + LRU |
| **存活性判别** | `can_allocate(key, tokens)` → 判断是否有足够空间（含可驱逐的） |

**与 KV APC 的关系：**

两者正交：

- **KV APC**：缓存"文本前缀"的 K/V 向量，按 token 块索引
- **EncoderCache**：缓存"视觉编码"的 embedding，按图像内容哈希索引

同一张图配上不同的 system prompt → 共享 ViT 输出（EncoderCache），但不共享图后文本的 KV。

### 4.4 Vision Encoder（ViT 模型）

**位置：** `src/multimodal/vit.py`

从零实现的 Qwen3 Vision Transformer（不与 HF 耦合），关键模块：

```
Qwen3VisionTransformer
├── patch_embed (Conv3d)          # 将 [T, C, H, W] 像素转为 patches（kernel=(2,16,16)）
├── pos_embed (nn.Embedding)      # 双线性插值位置编码
├── blocks × 24                   # Transformer blocks
│   └── Qwen3VisionBlock
│       ├── norm1 (LayerNorm)
│       ├── attn (Qwen3VisionAttention)
│       │   ├── qkv (Linear)
│       │   ├── RoPE (rotate_half)
│       │   └── ViTAttentionBackend (flash_attn_varlen_func)
│       ├── norm2 (LayerNorm)
│       └── mlp (Qwen3VisionMLP: fc1→GELU→fc2)
├── merger                        # 主路 merger（spatial merge: 将 2×2→1）
└── deepstack_merger_list         # DeepStack 层 merger（在指定 layer 输出处收集特征）
```

**ViT Attention 后端（flash_attn_varlen）：**

```1:53:src/layers/vision/vit_attn.py
from typing import List, Optional

import torch
from flash_attn import flash_attn_varlen_func


class ViTAttentionBackend:
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        **_kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.scale = head_dim**-0.5
        self._cu_seqlens: Optional[torch.Tensor] = None
        self._max_seqlen: int = 0

    def plan(self, cu_seqlens_cpu: torch.Tensor, seqlens: Optional[List[int]] = None) -> None:
        self._cu_seqlens = cu_seqlens_cpu.to(
            dtype=torch.int32, device=self.device, non_blocking=True
        )
        if seqlens:
            self._max_seqlen = max(seqlens) if seqlens else 0
        else:
            self._max_seqlen = int((cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item())

    def bind(self, cu_seqlens_gpu: torch.Tensor, max_seqlen: int) -> None:
        assert cu_seqlens_gpu.device.type == "cuda"
        assert cu_seqlens_gpu.dtype == torch.int32
        self._cu_seqlens = cu_seqlens_gpu
        self._max_seqlen = int(max_seqlen)

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        assert self._cu_seqlens is not None
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=self._cu_seqlens,
            cu_seqlens_k=self._cu_seqlens,
            max_seqlen_q=self._max_seqlen,
            max_seqlen_k=self._max_seqlen,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=False,
        )
```

多图合批时，`cu_seqlens` 定义了每张图的分段边界，一次性调用 flash-attn varlen，避免 per-image 的 SDPA 循环。

### 4.5 DeepStack（视觉特征注入 LM）

Qwen3-VL 的 DeepStack 机制：ViT 的不同层输出不同的视觉特征，注入到 LM 的前几层，增强 LM 对视觉信息的利用。

**实现方式：**

1. **ViT 侧**：在 `Qwen3VisionTransformer.forward()` 中，当遍历 `blocks` 时，如果当前层索引在 `deepstack_indexes` 中（如 `[8, 16, 24]`），通过对应的 `deepstack_merger` 将隐藏状态聚合并收集起来：

```416:424:src/multimodal/vit.py
        deepstack = []
        for i, blk in enumerate(self.blocks):
            hidden = blk(hidden, cos, sin, md["seqlens"], md["offsets"], attn)
            merger = self.deepstack_index_to_module.get(i)
            if merger is not None:
                deepstack.append(merger(hidden))

        merged = self.merger(hidden)
        self._deepstack = deepstack or None
```

2. **Encode 侧**：`_encode_visual()` 将 DeepStack 特征存储到 `MultiModalFeature.deepstack`。

3. **LM 侧**：`_build_deepstack()` 构造 `[ndps][N, H]` 的 DeepStack 缓冲区，LM forward 时在前 `ndps` 层做注入：

```127:130:src/models/qwen3.py
        for i, layer in enumerate(self.layers):
            h, residual = layer(positions, h, residual)
            if i < n_ds:
                h = h + deepstack_input_embeds[i]
```

### 4.6 MRoPE（3 轴旋转位置编码）

**这是多模态最核心的技术点之一。**

**为什么需要 MRoPE？**

纯文本只需要 1-D 位置编码（"第几个 token"）。但视觉 token 有 3 维结构：`(temporal, height, width)` —— 时间帧、高度方向、宽度方向。Qwen3-VL 使用 **interleaved MRoPE**：将 head_dim 分成三段，分别用三个轴的位置做旋转编码。

**两个关键模块：**

**a) 位置计算（CPU）：`_mrope.py` → `mrope_positions()`**

```4:9:src/multimodal/_mrope.py
def mrope_positions(
    input_ids: torch.Tensor,
    seqs: list,
    merge_size: int,
    device: torch.device,
) -> torch.Tensor:
```

输出 `[3, total_tokens]` 的 int64 张量，三行分别是 t/h/w 坐标。计算逻辑：

- 文本位置：`base + arange`（线性递增）
- 图像位置：替换为 `grid_slice(t, gh, gw)` 的 flatten 后的 (t, h, w)
- 图像结束后：从 `max(image_pos) + 1` 继续
- 记录 `mrope_next` 和 `rope_delta`（用于 chunked prefill 续接和 decode）

**b) RoPE 应用（GPU）：`mrope.py` → `MRotaryEmbedding`**

```52:68:src/layers/mrope.py
    def forward(self, positions, query, key):
        # 1-D → plain RoPE (decode / text); 3-D → MRoPE (MM prefill).
        if positions.ndim == 1 or positions.shape[0] < 3:
            pos = positions if positions.ndim == 1 else positions[0]
            fn = RotaryEmbedding._forward_cuda if query.is_cuda else RotaryEmbedding._forward_native
            return fn(self, pos, query, key)
        cos, sin = self._cos_sin(positions)
        if query.is_cuda:
            return _rope_cuda(
                query,
                key,
                cos.squeeze(1).contiguous(),
                sin.squeeze(1).contiguous(),
                self.head_size,
                self.rotary_dim,
            )
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)
```

Interleaved 模式：将 head_dim 按 `mrope_section`（如 `(24, 20, 20)`）分成三段，交错排布。每段 cos/sin 分别来自对应轴的位置。**1-D 位置时自动退化为 plain RoPE，复用同一个 Triton kernel。**

### 4.7 ViT CUDA Graph（预算式图捕获）

**位置：** `src/multimodal/vision.py` → `VisionEncoderCudaGraph`

**为什么需要？**ViT 的 forward 包含大量小 kernel（patch embed、bilinear pos embed、RoPE、LayerNorm、QKV、attention、MLP），CPU launch overhead 在 eager 模式下显著。CUDA Graph 可以一次性录制整个图，消除 launch 开销。

**预算式设计（Budget CG）：**

```166:170:src/multimodal/vision.py
class VisionEncoderCudaGraph:
    """Budget-based ViT CUDA Graph (subset of official EncoderCudaGraphManager).

    Captures power-of-two output-token budgets. Replay copies into static
    buffers then ``graph.replay()``. Falls back to eager when no budget fits.
    """
```

核心思想：

1. **捕获阶段**：为多个 2 的幂次 token budget（如 `[64, 128, 256, 512, 1024]`）分别捕获 CUDA Graph，每个 budget 用 dummy 输入填充到 `max_patches`
2. **回放阶段**：新图片的 total_out_tokens 找到最小能装下的 budget graph → 将真实数据 `copy_pad` 进静态 buffer → `graph.replay()`
3. **回退**：找不到匹配的 budget → eager forward

**与 Decode CG 的关系**：两者正交，可以同时启用。ViT CG 加速视觉编码，Decode CG 加速文本生成。

### 4.8 encode_and_merge（LM 侧编排）

**位置：** `src/models/qwen3_vl.py` → `Qwen3VLForCausalLM.encode_and_merge()`

这是 prefill 阶段的核心编排函数，将所有前面的组件串联起来：

```210:231:src/models/qwen3_vl.py
    def encode_and_merge(self, input_ids, seqs, device):
        """Prefill: ViT encode (window + cache) → scatter embeds → MRoPE positions."""
        text = self.vl_model.embed_tokens(input_ids)
        vis = self._encode_visual(seqs, device)
        embeds = text  # scatter in-place below; embed_tokens already returns a new tensor
        off = vi = 0
        for seq in seqs:
            cs, ce = seq.num_cached_tokens, seq.num_cached_tokens + seq.num_scheduled_tokens
            for mf in seq.mm_features:
                a, b = mf.start, mf.start + mf.num_encoder_tokens
                s, e = max(a, cs), min(b, ce)
                if s < e:
                    local = off + (s - cs)
                    src = vis[vi][s - a : e - a]
                    if src.dtype != embeds.dtype or src.device != embeds.device:
                        src = src.to(dtype=embeds.dtype, device=device, non_blocking=True)
                    embeds[local : local + (e - s)].copy_(src)
                vi += 1
            off += seq.num_scheduled_tokens
        positions = mrope_positions(input_ids, seqs, self.vision_tower.merge_size, device)
        self._mrope_module().ensure_cache_size(int(getattr(positions, "max_pos", 0)) + 1)
        return embeds, self._build_deepstack(seqs, embeds.shape[0], embeds.shape[1], device, embeds.dtype), positions
```

步骤：`embed_tokens(text)` → `_encode_visual(seqs)`（含 EncoderCache 查/写）→ scatter 视觉 embeds → `mrope_positions()` → `_build_deepstack()` → 返回 `(inputs_embeds, deepstack, positions_3d)`。

---

## 5. v3 与 v4 的代码差异详解

### 5.1 v3 的模块（文本纯栈）

```text
v3/src/
├── config.py                     # EngineConfig（text-only）
├── sampling_params.py
├── core/cuda_graph.py            # Decode CG
├── engine/
│   ├── engine.py                 # add_request(prompt) — 纯文本
│   ├── scheduler.py              # PD mix，无 encoder budget
│   ├── model_runner.py           # 无双模态分支
│   ├── input_builder.py          # 无 rope_delta
│   ├── sampler.py
│   └── model_inputs.py
├── models/
│   ├── qwen3.py                  # Qwen3ForCausalLM（无 DeepStack 注入点）
│   ├── loader.py
│   └── hub.py
├── layers/
│   ├── attention.py
│   ├── attention_backend.py
│   ├── activation.py
│   ├── layernorm.py
│   └── rotary.py                 # 仅 1-D RoPE
├── memory/
│   ├── sequence.py               # 无 MultiModalFeature，无 MM 字段
│   ├── block_manager.py
│   ├── kv_cache.py
│   ├── kv_ops.py
│   ├── manager.py                # 池维度用 config（无 text_config）
│   └── prefix/
│       ├── hash_index.py         # 无 cacheable 截断
│       ├── radix_index.py        # 无 cacheable 截断
│       ├── radix_tree.py
│       ├── hash_util.py
│       └── noop.py
└── utils/context.py
```

### 5.2 v4 新增的模块

| 模块 | 行数 | 作用 |
|------|------|------|
| `layers/mrope.py` | 87 | 交错 MRoPE（继承 RotaryEmbedding，复用 Triton kernel） |
| `layers/vision/vit_attn.py` | 53 | ViT 注意力后端（flash_attn_varlen） |
| `multimodal/processor.py` | 245 | HF AutoProcessor + 图像预处理缓存 |
| `multimodal/vit.py` | 434 | 完整 Qwen3 ViT 实现 |
| `multimodal/vision.py` | 455 | 视觉塔、EncoderCache、ViT CUDA Graph |
| `multimodal/_mrope.py` | 154 | LM 侧 3 轴 MRoPE 位置计算 |
| `multimodal/image.py` | 51 | PIL 图像缩放工具 |
| `multimodal/hasher.py` | 20 | SHA-256 图像哈希 |
| `models/qwen3_vl.py` | 325 | Qwen3-VL 编排（Attention/MoDEL/ForCausalLM + encode_and_merge + 权重加载） |

### 5.3 v3 被修改的模块

| 模块 | v3 行数 | v4 行数 | 变化内容 |
|------|--------|--------|---------|
| `models/qwen3.py` | ~155 | 209 | +`inputs_embeds` 参数、+`deepstack_input_embeds` 注入点 |
| `memory/sequence.py` | ~115 | 190 | +`MultiModalFeature` dataclass、+`mm_features`/`rope_delta`/`mrope_next`/`has_multimodal`/`first_mm_token`、+`prefix_cacheable_full_blocks()`/`reset_mrope_state()` |
| `engine/scheduler.py` | ~180 | 250 | +`encoder_compute_budget`、+`_clamp_chunk_for_encoder_budget()`、+preempt 时 `reset_mrope_state()` |
| `engine/engine.py` | ~250 | 376 | +`images=` 参数、+`_build_mm_features()`、+`_truncate_left()`（含 MM 特征重映射）、+多模态模型加载分支、+ViT CG 预暖 |
| `engine/model_runner.py` | ~210 | 347 | +多模态 prefill 分支（`encode_and_merge`）、+MRoPE cache 扩容、+`compile_prefill` 在 MM 时禁止 |
| `engine/input_builder.py` | ~135 | 209 | +decode 时 `rope_delta` 计算 |
| `memory/manager.py` | ~140 | 184 | +`text_config` 获取层数/头数、+sync 尊重 cacheable 长度 |
| `memory/prefix/hash_index.py` | ~80 | 104 | +match/publish 用 `prefix_cacheable_full_blocks()` |
| `memory/prefix/radix_index.py` | ~100 | 130 | 同 hash_index |
| `config.py` | ~50 | 88 | +`is_multimodal`、+`text_config`、+ViT CG 开关、+encoder_compute_budget |

---

## 6. 新技术点示例讲解

### 6.1 示例一：合批 ViT Attention

**问题：** v5 中多张图片的 ViT 是逐张 `SDPA` 循环执行的，TTFT 线性增长。

**v4 方案：** 使用 `flash_attn_varlen_func` 将多张图的 Q/K/V 拼接成 `[sum_tokens, H, D]`，通过 `cu_seqlens` 标定每张图的边界，一次性调用：

```39:52:src/layers/vision/vit_attn.py
    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        assert self._cu_seqlens is not None
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=self._cu_seqlens,
            cu_seqlens_k=self._cu_seqlens,
            max_seqlen_q=self._max_seqlen,
            max_seqlen_k=self._max_seqlen,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=False,
        )
```

`cu_seqlens` 在 `VisionTransformer.prepare_metadata()` 中根据 `grid_thw` 计算：

```329:336:src/multimodal/vit.py
        g = np.asarray(grids, dtype=np.int32)
        patches = g[:, 1] * g[:, 2]
        cu = np.concatenate(
            [np.zeros(1, dtype=np.int32), np.repeat(patches, g[:, 0]).cumsum(dtype=np.int32)]
        )
        if max_num_seqs is not None and (len(cu) - 1) < max_num_seqs:
            cu = np.concatenate(
                [cu, np.full(max_num_seqs - (len(cu) - 1), cu[-1], dtype=np.int32)]
            )
```

**效果：** 多图场景下 ViT forward 只跑一次，TTFT 显著降低。

### 6.2 示例二：EncoderCache 引用计数驱逐

**问题：** 简单 LRU 可能驱逐正在被 in-flight request 使用的 encoder 输出，导致逻辑不一致。

**v4 方案：** 引用计数 + token 预算驱逐：

```62:66:src/multimodal/vision.py
class EncoderCache:
    """Memory-aware encoder cache: token budget + refcount + LRU freeables.

    Aligns with official ``EncoderCacheManager`` (cache_size = embedding tokens),
    not entry-count LRU.
    """
```

`can_allocate()` 检查实际可释放空间（只算零引用 entry）：

```82:88:src/multimodal/vision.py
    def can_allocate(self, key: str, tokens: int) -> bool:
        if key in self._cache:
            return True
        freeable = sum(
            e.tokens for k, e in self._cache.items() if not self._refs.get(k)
        )
        return self.used_tokens - freeable + tokens <= self.max_tokens
```

`_evict_until()` 只驱逐零引用 entry：

```122:139:src/multimodal/vision.py
    def _evict_until(self, need: int) -> None:
        while self.used_tokens + need > self.max_tokens and self._cache:
            victim = None
            for k in self._cache:
                if not self._refs.get(k):
                    victim = k
                    break
            if victim is None:
                # Over-commit rather than drop in-flight entries.
                logger.warning(
                    "EncoderCache over-commit: used=%d need=%d max=%d",
                    self.used_tokens,
                    need,
                    self.max_tokens,
                )
                return
            e = self._cache.pop(victim)
            self.used_tokens -= e.tokens
            self._refs.pop(victim, None)
```

### 6.3 示例三：ViT CUDA Graph 预算式捕获与回放

**捕获：** 为每个 token_budget 创建 dummy 输入，执行一遍 forward，然后录制 CUDA Graph：

```256:267:src/multimodal/vision.py
            g = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(g, pool=self._graph_pool):
                out = self.vit(
                    buffers["pixel_values"],
                    metadata=meta,
                    concat_deepstack=True,
                    plan_attn=False,
                    split_images=False,
                )
                tmp.copy_(out)
```

**回放：** 将真实数据 `copy_pad` 进静态 buffer，然后 `graph.replay()`：

```339:342:src/multimodal/vision.py
        bg.graph.replay()
        self.hits += 1

        out = bg.output_buffer[:total]
```

**关键细节：** 捕获期间不调用 `plan_attn`（`plan_attn=False`），而是在 capture 前用 `attn.bind()` 预先绑定静态的 `cu_seqlens` GPU buffer。这样 CUDA Graph 中不包括动态的 `plan` 逻辑。

### 6.4 示例四：图前文本前缀缓存

**问题：** 多模态请求中，视觉 token 的 KV 不适合放进 token-hash 前缀索引（因为图像后的文本位置依赖图像内容）。

**v4 方案：** `Sequence.prefix_cacheable_full_blocks()` 将可缓存前缀截断到第一张图之前：

```156:160:src/memory/sequence.py
    def prefix_cacheable_full_blocks(self) -> int:
        n_full = self.num_full_blocks()
        if not self.mm_features:
            return n_full
        return min(n_full, self.first_mm_token // self.block_size)
```

`hash_index` 和 `radix_index` 在 match/publish 时都通过这个函数获取实际的 cacheable 范围，保证"含图及之后的页"不进 APC。

---

## 7. 关键不变量（v4 实现时必须遵守）

这些不变量来自 DESIGN.md，是 v4 多模态正确性的底线：

1. **未编码不得越过 image token。** Scheduler 中 `_clamp_chunk_for_encoder_budget()` 保证。
2. **含图页不得 publish/match 进 APC。** `prefix_cacheable_full_blocks()` 截断。
3. **Preempt 必须 reset MRoPE。** `scheduler.py` 的 `preempt()` 调用 `reset_mrope_state()`。
4. **KV 池用 text_config 层数/头。** `manager.py` 用 `text_config` 而非 `config`。
5. **Decode 不跑 ViT / 不加 DeepStack。** `model_runner.py` 中只有 prefill 时走 `encode_and_merge`。
6. **VL 不得关闭 FI Tensor Core / 破坏 decode CG 契约。** `attention_backend.py` 默认 `use_tensor_cores=True`。
7. **ViT CG capture 中不扩 MRoPE 表。** Capture 期间 MRoPE cache 大小固定。

---

## 8. 性能总结

| 场景 | v4 策略 | 预期表现 |
|------|--------|---------|
| 纯文本 decode | FlashInfer paged + Decode CUDA Graph | = v3（零回归） |
| 纯文本 prefill | FlashInfer prefill + torch.compile（可选） | = v3 |
| 单图 prefill | ViT flash-attn varlen + EncoderCache + VLM | 优于 per-image SDPA |
| 多图 prefill | Batch encode + 合批 attn | 显著优于逐张循环 |
| 同图二次请求 | EncoderCache hit → ViT=0 | 编码耗时趋零 |
| 重复 system prompt | KV APC（图前文本） | 前缀命中，跳过 prefill |

---

**本文基于 v4 源码（DESIGN.md + 75 个 .py 文件）编撰。建议配合 DESIGN.md 阅读设计决策，配合源码阅读实现细节。**
