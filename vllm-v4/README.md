# vllm-v4

主题：**在 v3 高性能文本栈上，完整实现 Qwen3-VL 多模态，并让 encode / decode / 算子路径都尽量高效。**

```text
v4 = v3（FlashInfer + Triton + Decode CG + hash/radix + PD mix）
   + Qwen3-VL 视觉塔 + DeepStack + 交错 MRoPE
   + EncoderCache（mm_hash → 视觉 embedding）
   + ViT 合批 Attention 用 FlashInfer ragged（+ 可选 ViT CG）
   + 图前文本 APC（hash 与 radix 都支持）
```

详细设计见 **[DESIGN.md](./DESIGN.md)**。

## 性能要点（相较 v5/v7）

- **ViT Attention**（`layers/vision/vit_attn.py`）：**flash-attn varlen**（原生支持
  Qwen3-VL `head_dim=72`）。LM 侧仍用 **FlashInfer paged**（连续批 + Decode CG /
  block table；flash-attn 无法直接替换该路径）。
  CUDA Graph capture 期间 ViT 回退 SDPA。环境变量只通过 `source .env.cuda.sh` 注入，
  **代码不耦合 CUDA/FlashInfer arch 环境**。
- **Decode 仍走 v3 的 FlashInfer paged + Decode CUDA Graph**，MRoPE 用 1-D 位置
  （`len-1 + rope_delta`）与文本共用同一条快路径、同一 Triton RoPE kernel。
- **EncoderCache** 命中直接复用视觉 embedding，跳过整座 ViT。
- **图前文本 APC**：`Sequence.prefix_cacheable_full_blocks()` 把可缓存前缀截到
  第一张图之前，hash / radix 后端都遵守，图像 KV 绝不进 token-hash 索引。

## 目录

```text
src/
  config.py                     # + is_multimodal / text_config / ViT-CG 开关
  layers/
    mrope.py                    # 交错 MRoPE（复用 v3 Triton RoPE kernel）
    vision/vit_attn.py          # ★ FlashInfer ragged ViT attention 后端
  multimodal/
    processor.py                # HF AutoProcessor → ids + pixels + spans
    vit.py                      # Qwen3 ViT + merger + DeepStack（FlashInfer）
    vision.py                   # Qwen3VisionTower + LRU EncoderCache
    _mrope.py                   # LM 侧 3 轴 MRoPE 位置（prepare-once）
  models/
    qwen3.py                    # LM forward 扩展 inputs_embeds / deepstack
    qwen3_vl.py                 # 塔 + LM 编排：encode_and_merge / 权重加载
  core/vit_cuda_graph.py        # budget-based ViT CUDA Graph runner
  engine/                       # scheduler(preempt reset) / runner(MM prefill) / engine(images)
  memory/                       # + MultiModalFeature，前缀后端图前截断
```

## 用法

```bash
# 文本（与 v3 完全一致）
python run.py --model Qwen3-0.6B --prompt "Hello"

# 多模态（Qwen3-VL）
python run.py --model Qwen3-VL-2B-Instruct --image cat.jpg --prompt "描述这张图" --compile
python run.py --model Qwen3-VL-2B-Instruct --image cat.jpg --enable-vit-cg
```

## 与其它版本

| 版本 | 关系 |
| --- | --- |
| v3 | **底座**：控制面与 LM 数据面原样继承，文本路径零回退（`tests/` 全绿）|
| v5 | 提供 ViT / Qwen3-VL / DeepStack / MRoPE **零件**；v4 用 FlashInfer ragged 取代其 per-image SDPA |
| v7 | 提供图前前缀、prepare-once 等 **合同** |
| 官方 vLLM | EncoderCache、`vit_attn_wrappers`、Encoder CG **对标** |

## 测试

```bash
python -m pytest tests/ -q     # 文本前缀/调度 + 多模态前缀截断/MRoPE 位置（CPU 即可）
```
