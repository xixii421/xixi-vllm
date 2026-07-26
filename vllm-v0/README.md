# vllm-v0 — 最小自回归推理

> 从零实现 Qwen3 单请求自回归推理：完整模型骨架、SDPA Attention、可选采样策略。

---

## 相对于无（起点）

| 能力 | 说明 |
|------|------|
| Qwen3 模型骨架 | RMSNorm → QKV proj → RoPE → Attention → SwiGLU FFN |
| 权重加载 | safetensors 流式读取 + HuggingFace config |
| SDPA Attention | `F.scaled_dot_product_attention`，因果 mask |
| 采样策略 | temperature / top-k / top-p / min-p |
| 流式输出 | `generate()` 逐 token yield |

**限制：** 无 KV Cache，每步重算全部历史 token。

---

## 快速开始

```bash
cd vllm-v0

# 下载模型（需先修改 src/model_hub.py 中的缓存路径）
python run.py --download

# 推理（默认从 config.yaml 读取参数）
python run.py --model Qwen3-0.6B --prompt "Hello, my name is"

# 列出可用模型
python run.py --list
```

---

## 核心源码

| 文件 | 内容 |
|------|------|
| `src/engine.py` | Engine：`encode → forward → sample → yield` |
| `src/model.py` | Qwen3 骨架 + 层堆叠 + `forward()` |
| `src/model_loader.py` | safetensors 权重装载 |
| `src/sampler.py` | logits → token（temperature / top-k / top-p） |
| `src/config.py` | EngineConfig（model / dtype / enforce_eager） |
| `src/layers/layernorm.py` | RMSNorm |
| `src/layers/rotary.py` | RoPE 位置编码 |
| `src/layers/attention.py` | GQA + SDPA |
| `src/layers/activation.py` | SiluAndMul（SwiGLU） |

---

## 文档

| 顺序 | 文档 | 弄清什么 |
|:--:|------|---------|
| 1 | [architecture.md](docs/architecture.md) | 自回归循环全貌 |
| 2 | [layers.md](docs/layers.md) | Attention / MLP / Decoder 数据流 |
| 3 | [layers/rmsnorm.md](docs/layers/rmsnorm.md) | Pre-Norm + fused residual |
| 4 | [layers/rotary.md](docs/layers/rotary.md) | 位置如何旋进 Q/K |
| 5 | [layers/attention.md](docs/layers/attention.md) | GQA + 因果 SDPA |
| 6 | [layers/activation.md](docs/layers/activation.md) | SwiGLU 的 SiLU 门控 |
| 7 | [model.md](docs/model.md) | 层堆叠 + 权重装载 |
| 8 | [sampler.md](docs/sampler.md) | 采样策略实现 |
| 9 | [engine.md](docs/engine.md) | encode → forward → sample 编排 |
