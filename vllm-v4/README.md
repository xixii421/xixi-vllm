# vllm-v4 — 多模态（Qwen3-VL）

> 在 v3 高性能文本栈上完整实现 Qwen3-VL：ViT 视觉编码器 + DeepStack + 交错 MRoPE + EncoderCache + ViT CUDA Graph。

---

## 相对 v3 新增

| 新增能力 | 对应模块 | 说明 |
|---------|---------|------|
| Qwen3-VL 模型 | `src/models/qwen3_vl.py` | ViT + merger + LM 编排 |
| ViT Attention | `src/layers/vision/vit_attn.py` | `flash_attn_varlen_func`（非标准 head_dim=72） |
| DeepStack | `src/multimodal/vit.py` | 多层 visual features 注入 LM |
| 交错 MRoPE | `src/layers/mrope.py` + `src/multimodal/_mrope.py` | 3 轴位置编码，图像和文本 token 混合 |
| EncoderCache | `src/multimodal/vision.py` | mm_hash → 视觉 embedding，命中跳过 ViT |
| ViT CUDA Graph | `src/core/vit_cuda_graph.py` | budget-based CG，可选 `--enable-vit-cg` |
| 图前文本 APC | `src/memory/sequence.py` | prefix_cacheable_full_blocks 截到第一张图前 |
| 多模态推理入口 | `run.py` `--image` | `run_image()` 支持图片输入 |
| compile prefill | `config.py` `compile_prefill` | CG 开启时自动 torch.compile prefill |

**LM 文本路径零回退：** v3 的 FlashInfer + Decode CG + Triton 全部继承。

---

## 快速开始

```bash
cd vllm-v4

# 文本推理（与 v3 完全一致）
python run.py --model Qwen3-0.6B

# 多模态推理
python run.py --model Qwen3-VL-2B-Instruct --image cat.jpg --prompt "描述这张图"

# 启用 ViT CUDA Graph
python run.py --model Qwen3-VL-2B-Instruct --image cat.jpg --compile --enable-vit-cg

# 测试
python -m pytest tests/ -q
```

---

## 核心源码

| 文件 | v3→v4 |
|------|-------|
| `src/models/qwen3_vl.py` | **新增** — Qwen3-VL 模型编排（塔 + LM） |
| `src/models/qwen3.py` | **改写** — LM forward 扩展 inputs_embeds / deepstack |
| `src/multimodal/processor.py` | **新增** — HF processor → ids + pixels + spans |
| `src/multimodal/vit.py` | **新增** — ViT + merger + DeepStack |
| `src/multimodal/vision.py` | **新增** — VisionTower + LRU EncoderCache |
| `src/multimodal/_mrope.py` | **新增** — 3 轴 MRoPE 位置编码 |
| `src/layers/vision/vit_attn.py` | **新增** — flash-attn varlen ViT attention |
| `src/layers/mrope.py` | **新增** — 交错 MRoPE（复用 v3 Triton RoPE） |
| `src/core/vit_cuda_graph.py` | **新增** — budget-based ViT CG |
| `src/config.py` | **扩展** — is_multimodal / vit_cg / compile_prefill / encoder_budget |
| `src/memory/sequence.py` | **改写** — prefix_cacheable_full_blocks 图前截断 |

---

## 文档

| 文档 | 说明 |
|------|------|
| [architecture.md](docs/architecture.md) | v4 多模态全架构（ViT + DeepStack + MRoPE + EncoderCache） |
