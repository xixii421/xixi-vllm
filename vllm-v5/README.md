# vllm-v5

主题：**多模态（Qwen3-VL）** — 在 v4（FlashInfer + Continuous Batching + CUDA Graph）之上接上 ViT / MRoPE / DeepStack。

## 模型支持范围

| 家族 | 状态 | 说明 |
| --- | --- | --- |
| **Qwen3-VL** | **本版实现** | 完整路径：`MultiModalProcessor` → ViT → merge embeds → MRoPE prefill → 文本 decode |
| Qwen2-VL | 预留注册 | 结构相近，可后续复用 |
| LLaVA / InternVL 等 | **不做** | ViT/Projector/RoPE 差异大；见 `src/multimodal/registry.py` stub |

教学原则：**只做一条能跑通且可测的多模态路径**，不要假装「一套代码吃掉所有 VLM」。

## 架构增量（相对 v4）

```text
src/multimodal/
  processor.py     HF AutoProcessor → token_ids + pixel_values + span info
  vit.py / vision.py  Qwen3 Vision Tower + EncoderCache
  _mrope.py        3D positions for image patches
  registry.py      已实现 / 规划中的 VLM 列表
src/layers/mrope.py          MRotaryEmbedding
src/models/qwen3_vl.py       VisionTower + VL LM + encode_and_merge
src/engine/engine.py         add_request(..., images=[PIL...])
src/engine/model_runner.py   prefill 时 embeds + deepstack；KV/CG 用 text_config
```

数据流：

1. `images` → Processor 把 `<image>` placeholder 展开成 visual token 序列  
2. Prefill：ViT 编码 → 写入 `inputs_embeds` 对应 span → DeepStack 注入浅层 → MRoPE  
3. Decode：与 v4 相同（visual 已进 KV）；可选 CUDA Graph（1D position）

## 快速开始

```bash
cd vllm-v5
bash setup_env.sh
source .venv/bin/activate
pip install -r requirements.txt   # 含 Pillow / qwen-vl-utils

# 文本（与 v4 兼容，Qwen3）
python run.py --model ~/huggingface/Qwen3-0.6B --temperature 0 --max-new-tokens 64

# 多模态（Qwen3-VL）
python run.py --model ~/huggingface/Qwen3-VL-2B-Instruct \
  --image /path/to/photo.jpg \
  --prompt "Describe this image in one sentence." \
  --temperature 0 --max-new-tokens 64 --eager
```

建议多模态首次用 `--eager`（关闭 Decode CUDA Graph），确认图文正确后再开 `--compile`。

### ViT CUDA Graph（可选）

对齐 vLLM encoder cudagraph / dzyy：按 visual-token budget 捕获 ViT 整图前向（含 DeepStack 输出缓冲）。

```python
EngineConfig(..., enforce_eager=True, enable_vit_cuda_graph=True)
```

限制：当前仅对 **base grid `[1,16,16]`（64 visual tokens）** 的图像走 CG，其它分辨率回退 eager。

性能对比（4090，Qwen3-VL-2B，256×256 图）：

```bash
PYTHONPATH=. python bench/bench_vit_cg.py --model ~/huggingface/Qwen3-VL-2B-Instruct
```

报告见 `bench/results/vit_cg_bench.md`。

## 环境

- GPU：单卡（教程路径）；`CUDA_DEVICE_ORDER=PCI_BUS_ID` 在多卡机器上选定目标卡  
- 依赖：FlashInfer（同 v4）、`transformers` 支持 Qwen3-VL、Pillow  
- 推荐权重：`~/huggingface/Qwen3-VL-2B-Instruct`

## 相对 v4 的不做项

- TP / PP / 多机（留给 v6）  
- LLaVA / InternVL 全路径  
- ViT CUDA Graph / DP 切分（dzyy_vllm 有；教学版单卡即可）
