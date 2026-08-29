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
    <img src="https://img.shields.io/badge/v4_TP--PP--DP-gray?style=flat-square&label">
    <img src="https://img.shields.io/badge/v5_MoE--VL-gray?style=flat-square&label">
  </p>

  <p>从零实现的 Qwen3 推理引擎，按 v0–v5 冻结每一阶段。迭代目录见 <a href="项目目录.html">项目目录.html</a>。</p>

  
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

> 以torch2.13.0版本为准

```bash
uv pip install torch==2.13.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

uv pip install flashinfer-python flashinfer-cubin \
    --index-url https://flashinfer.ai/whl \
    --extra-index-url https://pypi.org/simple

uv pip install flashinfer-jit-cache \
    --index-url https://flashinfer.ai/whl/cu130

uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.47/flash_attn-2.8.3+cu130torch2.13-cp312-cp312-linux_x86_64.whl

uv pip install transformers huggingface_hub \
    safetensors tqdm numpy Pillow pyyaml einops xxhash
```

### 4. Qwen3 模型下载

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

# v2 — Continuous Batching
PYTHONPATH=vllm-v2 python vllm-v2/run.py

# v3 — FlashInfer + Decode CUDA Graph
PYTHONPATH=vllm-v3 python vllm-v3/run.py --compile

# v4 — 张量 / 流水线 / 数据并行
PYTHONPATH=vllm-v4 python vllm-v4/run.py --model Qwen3-0.6B --tp 1

# v5 — MoE / 视觉 / OpenAI API（文档仍在补）
# 见 vllm-v5/README.md；单卡路径已能跑
```


## v0～v5 各版本 README

| 版本 | 标题 | 核心技术 |
|------|------|---------|
| [vllm-v0](vllm-v0/README.md) | 最小自回归推理 | 裸模型加载、Naive 逐 token 生成 |
| [vllm-v1](vllm-v1/README.md) | KV Cache + PagedAttention + 前缀缓存 | KV Cache 复用、PagedAttention 显存管理、Hash/Radix 前缀缓存 |
| [vllm-v2](vllm-v2/README.md) | Continuous Batching | 动态组批、三面架构（Scheduler/KVManager/ModelRunner）、torch.compile |
| [vllm-v3](vllm-v3/README.md) | 算子级性能提升 | FlashAttention paged attention、Decode CUDA Graph、`F.rms_norm`、Gumbel-Max 采样 |
| [vllm-v4](vllm-v4/README.md) | 多卡并行 | DP / TP / PP，V1 式 EngineCore |
| [vllm-v5](vllm-v5/README.md) | MoE、视觉、API | Qwen3-MoE / Qwen3-VL、AsyncLLM、OpenAI HTTP（文档待完善） |
| [vllm-v6]()| 量化 | 计划中 | 