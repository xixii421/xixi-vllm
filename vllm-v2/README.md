# vllm-v2 — Continuous Batching

在 **v1**（分页 KV + PagedAttention 契约 + hash/radix 前缀）之上加入多请求调度。

**设计说明（推荐阅读）**：[docs/00-V2版本设计解释.md](docs/00-V2版本设计解释.md)（CB 动机） · [docs/02-代码架构与三面分离.md](docs/02-代码架构与三面分离.md)（源码怎么拆） · [docs/01-相比v1的增改.md](docs/01-相比v1的增改.md)

| 平面 | 职责 | 路径 |
| --- | --- | --- |
| **控制面** | waiting/running、chunked prefill、PD 混跑、preempt | `src/engine/scheduler.py` |
| **数据面** | 组 batch、绑 `ForwardContext`、跑模型、采样 | `src/engine/model_runner.py` |
| **计算面** | Attention / PagedStore / Qwen3 / Sampler | `src/layers/` · `src/pagedattention/` · `src/model*.py` |

`Engine` 只做组装：`step = schedule → run → postprocess`。

## 快速开始

```bash
cd vllm-v2
source ../.venv/bin/activate   

python run.py --model Qwen3-0.6B --temperature 0 --max-new-tokens 32
python run.py --model Qwen3-0.6B --batch --max-new-tokens 16

python -m bench.compare --model Qwen3-0.6B
python -m bench.compare --model Qwen3-0.6B --batch-size 4 --max-new 32,64 --skip-prefix

# Chunked P+D 混跑：看的是公平性（dec@P / short_P），不是总吞吐
python -m bench.compare_chunked --model Qwen3-0.6B --budget 32 --long-prompt 512 --conc 2,4
python -m bench.compare_chunked --model Qwen3-0.6B --skip-uniform --budget 32 --long-prompt 512
```

## 布局

```text
src/
  config.py / sampling_params.py
  kv/                 # 块池 + 前缀（与 v1 对齐命名）+ 多请求 Sequence
  pagedattention/     # slots + PagedStore（公开: slots_tensor / PagedStore）
  layers/             # Attention 支持 Decode 多请求
  model*.py sampler   # 计算面
  engine/
    scheduler.py      # 控制面
    model_runner.py   # 数据面
    engine.py         # 门面（含 reset_kv）
bench/                # v0/v1/v2 对比
```



