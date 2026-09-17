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

## uv 环境配置

### 1. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version  # 验证
```

### 2. 同步项目环境

```bash
uv sync --locked
```

这条命令会根据 `.python-version` 使用 Python 3.12，创建根目录下的 `.venv`，并严格按 `uv.lock` 安装依赖。无需手动激活虚拟环境，后续命令可直接用 `uv run`。

> 当前锁定环境面向 Linux x86_64，需要支持 CUDA 13.0 的 NVIDIA 驱动。

`pyproject.toml` 是 uv 环境的权威依赖入口：PyTorch 2.13 使用 CUDA 13.0 官方 index，FlashInfer 0.6.15 使用官方 index，FlashAttention 使用与 Python 3.12 / CUDA 13.0 / PyTorch 2.13 对齐的预编译 wheel，避免本地长时间 CUDA 编译。FlashInfer 额外的 JIT cache wheel 约 1.5 GiB，不是运行必需项，因此没有纳入默认环境。

可选依赖组：

```bash
# 生成文档插图
uv sync --locked --group docs

# 使用 ModelScope 下载模型
uv sync --locked --group modelscope
```

### 3. Qwen3 模型下载

项目内置模型下载逻辑（`src/model_hub.py` / `utils/model_hub.py`），运行推理前需先将权重下载到本地。
**手动下载到指定目录：**

```bash
# 方式一：Hugging Face（公开模型无需登录）
uv run hf download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B

# 方式二：modelscope（国内更稳定）
uv sync --locked --group modelscope
uv run python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', cache_dir='~/huggingface')"
```

**然后用本地路径运行（跳过自动下载）：**

```bash
PYTHONPATH=vllm-v3 uv run python vllm-v3/run.py --model ~/huggingface/Qwen3-0.6B
```

> 默认缓存目录为 `~/huggingface/`。Qwen3 模型名（repo_id）见各版本代码中的 `QWEN3_REGISTRY`。


## 运行各版本

```bash
# v0 — 基础自回归生成
PYTHONPATH=vllm-v0 uv run python vllm-v0/run.py

# v1 — KV Cache
PYTHONPATH=vllm-v1 uv run python vllm-v1/run.py

# v2 — Continuous Batching
PYTHONPATH=vllm-v2 uv run python vllm-v2/run.py

# v3 — FlashInfer + Decode CUDA Graph
PYTHONPATH=vllm-v3 uv run python vllm-v3/run.py --compile

# v4 — 张量 / 流水线 / 数据并行
PYTHONPATH=vllm-v4 uv run python vllm-v4/run.py --model Qwen3-0.6B --tp 1

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
