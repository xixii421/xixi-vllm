# vllm-v5 — 生产级推理引擎架构

> 面向生产环境的完整推理引擎：多 GPU 分布式、模块化 runtime、FP8/FP4 量化、MLA/DeepSeek 支持。

---

## 相对 v4 新增

| 新增能力 | 对应模块 | 说明 |
|---------|---------|------|
| 多 GPU 分布式 | `src/distributed/` | tensor/pipeline/expert parallelism |
| 模块化 runtime | `src/runtime/` | runner / input / execute / vision 分层 |
| 配置系统 | `src/config/` | 多文件配置（model / engine / server / parallel） |
| MoE 支持 | `src/layers/moe/` | expert routing + load balancing |
| FP8/FP4 量化 | 各层量化 kernel | 低精度 GEMM + attention |
| MLA (Multi-Latent Attention) | `src/layers/` | DeepSeek-V2/V3 风格注意力 |
| Benchmark 框架 | `bench/lib/` + `bench/suites/` | 完整的微基准 + 端到端对比套件 |
| 调度层 | `src/scheduling/` | 更精细的 scheduler，prefix 独立模块 |

---

## 快速开始

```bash
cd vllm-v5

# 公共 API
from src import Engine, EngineConfig, SamplingParams

# 微基准
PYTHONPATH=. python -m bench.run --suite e2e,ttft,decode

# 单元测试
pytest tests/unit -q
pytest tests/gpu -q
pytest tests/distributed -q
```

---

## 目录

```text
src/
  config/          配置系统（model / engine / server / parallel）
  scheduling/      调度 + prefix 管理
  runtime/         模块化推理运行时
    runner/        ModelRunner 实现
    input/         输入处理
    execute/       forward 执行
    vision/        多模态处理
  layers/          算子层（attention / norm / activation / moe）
  models/          模型实现
  cuda_graph/      CUDA Graph 管理
  distributed/     分布式通信
  sampling/        采样
  utils/           工具函数
```

---

## 文档

| 文档 | 说明 |
|------|------|
| [docs/README.md](docs/README.md) | 架构总览（控制/计算分离、调度流、计算流） |
