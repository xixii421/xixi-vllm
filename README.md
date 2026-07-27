<div align="center">
  <img src="assets/banner1.png" alt="dzyy-vllm" width="960">

  <p>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.13-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.13"></a>
    <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-13.0-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA 13.0"></a>
    <a href="https://flashinfer.ai/"><img src="https://img.shields.io/badge/FlashInfer-0.6.15-6366f1?style=flat-square" alt="FlashInfer"></a>
    <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/lint-ruff-7c3aed?style=flat-square" alt="Ruff"></a>

  </p>

  <p>
    <img src="https://img.shields.io/badge/v0_minimal-gray?style=flat-square&label">
    <img src="https://img.shields.io/badge/v1_kv--cache-gray?style=flat-square&label">
    <img src="https://img.shields.io/badge/v2_continuous--batching-gray?style=flat-square&label">
    <img src="https://img.shields.io/badge/v3_flashinfer-gray?style=flat-square&label">
    <img src="https://img.shields.io/badge/v4_multimodal-gray?style=flat-square&label">
    <img src="https://img.shields.io/badge/v5_distributed-gray?style=flat-square&label">
  </p>

  
</div>

---

## uv 快速配置指南

### 1. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version  # 验证
```

### 2. 创建虚拟环境

```bash
# 方案 A：Python 3.12
uv venv .venv-py312 --python 3.12
source .venv-py312/bin/activate
```

### 3. 安装 PyTorch + FlashInfer + FlashAttention

官方 `pip install flash-attn` 会触发很久 CUDA 源码编译，并且还会把CPU编译打满，使用 FlashInfer 官方 index + 社区 FlashAttention 预编译 wheel成本更低，但是同样需要版本对齐。

```bash
# PyTorch 2.13
uv pip install torch==2.13.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# FlashInfer 三合一（零编译）
uv pip install flashinfer-python flashinfer-cubin \
    --index-url https://flashinfer.ai/whl

# 非必要
uv pip install flashinfer-jit-cache \
    --index-url https://flashinfer.ai/whl/cu130

# FlashAttention 预编译 wheel（Linux x86_64，torch 2.13 + cu130）
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.47/flash_attn-2.8.3+cu130torch2.13-cp312-cp312-linux_x86_64.whl
```

### 4. 安装其余依赖

```bash
uv pip install transformers huggingface_hub \
    safetensors tqdm numpy Pillow pyyaml einops xxhash
```

### 5. Qwen3 模型下载

项目内置自动下载逻辑（`src/model_hub.py` / `src/models/hub.py`），首次运行时从 HuggingFace 拉取权重。
**手动下载到指定目录：**

```bash
# 方式一：huggingface-cli，需要登录
hf download Qwen/Qwen3-1.7B --local-dir ~/huggingface/Qwen3-1.7B

# 方式二：modelscope（国内更稳定）
uv pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B', cache_dir='~/huggingface')"
```

**然后用本地路径运行（跳过自动下载）：**

```bash
PYTHONPATH=vllm-v3 python vllm-v3/run.py --model ~/huggingface/Qwen3-1.7B
```

> 默认缓存目录为 `~/huggingface/`。Qwen3 模型名（repo_id）见各版本代码中的 `QWEN3_REGISTRY`。


## 运行各版本

```bash
# v0 — 基础自回归生成
PYTHONPATH=vllm-v0 python vllm-v0/run.py

# v1 — KV Cache
PYTHONPATH=vllm-v1 python vllm-v1/run.py

# v2 — Continuous Batching + torch.compile
PYTHONPATH=vllm-v2 python vllm-v2/run.py

# v3 — FlashInfer + Decode CUDA Graph
PYTHONPATH=vllm-v3 python vllm-v3/run.py --compile

# v4 — 多模态 (Qwen3-VL)
PYTHONPATH=vllm-v4 python vllm-v4/run.py \
    --model Qwen3-VL-2B-Instruct --image ./vllm-v4/cat.jpg --prompt "描述这张图"

# v5 - MoE 待定

# 鉴于代码量与优化量太大，还在完善中，目前代码已经可以成功运行
```


## v0～v5 各版本 README

| 版本 | 标题 | 核心技术 |
|------|------|---------|
| [vllm-v0](vllm-v0/README.md) | 最小自回归推理 | 裸模型加载、Naive 逐 token 生成 |
| [vllm-v1](vllm-v1/README.md) | KV Cache + PagedAttention + 前缀缓存 | KV Cache 复用、PagedAttention 显存管理、Hash/Radix 前缀缓存 |
| [vllm-v2](vllm-v2/README.md) | Continuous Batching | 动态组批、三面架构（Scheduler/KVManager/ModelRunner）、torch.compile |
| [vllm-v3](vllm-v3/README.md) | 算子级性能提升 | FlashInfer paged attention、Decode CUDA Graph、Triton RMSNorm、Gumbel-Max 采样 |
| [vllm-v4](vllm-v4/README.md) | 多模态（Qwen3-VL） | DeepStack、MRoPE、EncoderCache、ViT CUDA Graph、Vision Token 管理 |
| [vllm-v5](vllm-v5/README.md) | TP、PP、MoE | 文档待完善 |
| [vllm-v6]()| 若干量化方案 | 计划中 | 