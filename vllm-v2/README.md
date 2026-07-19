# vllm-v2 — Continuous Batching（基于 v1）

在 **v1**（分页 KV + PagedAttention 契约 + hash/radix 前缀）之上加入多请求调度。  
主体计算/存储代码复用 v1；批处理控制流参考原 v3，并按 **三面分离** 重组。

| 平面 | 职责 | 路径 |
| --- | --- | --- |
| **控制面** | waiting/running、Prefill-first、preempt | `src/engine/scheduler.py` |
| **数据面** | 组 batch、绑 `ForwardContext`、跑模型、采样 | `src/engine/model_runner.py` |
| **计算面** | Attention / PagedStore / Qwen3 / Sampler | `src/layers/` · `src/pagedattention/` · `src/model*.py` |

`Engine` 只做组装：`step = schedule → run → postprocess`。

## 快速开始

```bash
cd vllm-v2
source ../.venv/bin/activate   # 或 vllm-v0/.venv

python run.py --model Qwen3-0.6B --temperature 0 --max-new-tokens 32
python run.py --model Qwen3-0.6B --batch --max-new-tokens 16

python test/test_scheduler.py
python test/test_pagedattention.py
python test/test_e2e_batch.py --model Qwen3-0.6B
```

## 布局

```text
src/
  config.py / sampling_params.py
  kv/                 # 块池 + 前缀（v1）+ 多请求 Sequence
  pagedattention/     # slots + PagedStore（v1，write 支持 B>1）
  layers/             # Attention 支持 Decode 多请求
  model*.py sampler   # 计算面
  engine/
    scheduler.py      # 控制面
    model_runner.py   # 数据面
    engine.py         # 门面
```

设计说明见 [docs/architecture.md](docs/architecture.md)。原 `vllm-v3/` 仍保留作对照，本版为推荐入口。
