<div align="center">
  <img src="assets/banner1.png" alt="xixi-vllm" width="960">
  <p>从零实现的 Qwen3 推理引擎，仓库仅维护 v4：单卡推理与 TP / PP / DP 多卡并行。</p>
</div>

## 能力与目录

支持 Paged KV Cache、Hash / Radix 前缀缓存、Continuous Batching、FlashAttention、FlashInfer 算子、Decode CUDA Graph 和 prefill `torch.compile`。控制面采用 `Engine → Client → EngineCore → Executor → Worker`，单卡和多卡使用同一套接口。

```text
.
├── src/                   # v4 源码；公共工具已合并进模块内部
│   ├── control/           # 调度与多进程控制面
│   ├── data/              # 序列、KV 账本与前缀缓存
│   ├── compute/           # 模型、权重加载与并行算子
│   ├── distributed/       # 进程组与通信
│   ├── utils/             # 运行辅助工具
│   ├── model_hub.py       # 模型路径与下载
│   └── sampling_params.py # 采样参数
├── run.py                 # 单卡及本机 TP / PP / DP 入口
├── run_torchrun.py        # 跨机 TP / PP 入口
├── config.yaml            # 推理与并行配置
├── test/                  # 正确性测试脚本
├── bench/                 # 基准测试与硬件探测
├── docs/                  # 并行设计、硬件互联与实测文档
├── assets/                # README 图片
├── pyproject.toml         # 环境依赖
└── uv.lock                # 依赖锁定
```

实现和参数说明见 [v4 README](docs/README.md)，性能数据见 [性能对比](docs/03-性能对比.md)。

## 环境配置

本项目使用本机 WSL2（Ubuntu）运行。以下命令均在 WSL 终端的仓库根目录执行；锁定环境面向 Linux x86_64、Python 3.12 和 CUDA 13.0。

Windows 侧安装支持 WSL2 的 NVIDIA 驱动，并先在 WSL 中运行 `nvidia-smi` 确认 GPU 可见。可通过 `/mnt/c/...` 或 `/mnt/d/...` 进入 Windows 上的仓库；为提高文件读写速度，也可将仓库放在 WSL 的 `~/projects/` 下。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --locked

# 可选：生成文档插图 / 使用 ModelScope 下载模型
uv sync --locked --group docs
uv sync --locked --group modelscope
```

在 WSL 中创建和使用根目录 `.venv`，后续命令使用 `uv run`；该环境不能与 Windows Python 的虚拟环境混用。PyTorch、FlashInfer 和 FlashAttention 的来源与版本由 `pyproject.toml` / `uv.lock` 指定。

## 模型准备

```bash
uv run hf download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B
```

也可使用 ModelScope，将模型下载到同一目录：

```bash
uv sync --locked --group modelscope
uv run python -c "from pathlib import Path; from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir=str(Path.home() / 'huggingface' / 'Qwen3-0.6B'))"
```

模型名默认解析到 `~/huggingface/<模型名>`，也可通过 `--model` 传入本地权重目录。并行测试使用的大模型需先单独下载。

## 运行

以下命令在仓库根目录执行：

```bash
# 单卡；添加 --cg / --compile 启用优化
uv run python run.py --model ~/huggingface/Qwen3-0.6B
uv run python run.py --model Qwen3-0.6B --cg --compile

# 张量并行 / 流水线并行 / 数据并行（需要相应数量的 GPU）
uv run python run.py --model Qwen3-0.6B --tp 2 --cg
uv run python run.py --model Qwen3-8B --pp 2 --cg
uv run python run.py --model Qwen3-0.6B --dp 2 --cg --batch

# torchrun 只用于 TP / PP，不与 --dp 混用
uv run torchrun --nproc_per_node=2 run_torchrun.py --tp 2 --pp 1
```

本机总 GPU 数为 `dp × tp × pp`。完整启动方式和不均分 PP 配置见 [v4 README](docs/README.md)。

## 测试与基准

v4 的测试通过独立脚本运行，测试脚本自行初始化进程组，不使用 pytest 收集：

```bash
# 不需要模型权重的分片与通信测试；仍需完整项目依赖
uv run python test/test_sharding.py
uv run python test/test_pp_comm.py

# 需要本地模型和 CUDA GPU
uv run python test/test_parallel.py --tp 1 --pp 1
uv run python test/test_parallel.py --tp 2 --pp 1
uv run python test/test_dp.py --dp 2 --n-reqs 4

# v4 吞吐基准
uv run python bench/bench_v4.py --tp 1 --cg
```

更多基准用例、硬件探测和官方 vLLM 对照见 [bench/README.md](bench/README.md)。官方 vLLM 对照使用单独的 Python 环境。
