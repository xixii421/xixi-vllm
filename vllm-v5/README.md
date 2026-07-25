# vllm-v7

教学向连续批处理引擎：控制面（Engine / Scheduler）与计算面（ModelRunner / layers）分离。

## 文档

- 架构：[`docs/README.md`](docs/README.md)（控制/计算、调度流、计算流）
- 设计约定：[`DESIGN.md`](DESIGN.md)

## 入口

```bash
# 公共 API
from src import Engine, EngineConfig, SamplingParams

# 微基准（bench/lib）
PYTHONPATH=. python -m bench.run --suite e2e,ttft,decode

# vs 官方 vLLM（CG 矩阵）
PYTHONPATH=. python scripts/bench_vs_vllm_full.py --mode cg --suite text

# 模型下载辅助
PYTHONPATH=. python scripts/hub.py --help
```

## 测试

```bash
pytest tests/unit -q          # CPU 快测
pytest tests/gpu -q           # 单卡
pytest tests/distributed -q   # 多卡
```

## 目录

```text
src/           引擎核心
docs/          架构讲义
bench/lib/     微基准核心库
scripts/       少量 CLI（对比 / hub / 分片）
tests/         精简 unit / gpu / distributed
```
