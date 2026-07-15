# vllm-v3

主题：**调度与 Continuous Batching**（纯 PyTorch）。

在 v2 PagedAttention 之上拆出：

- `Scheduler`：waiting / running、Prefill-first、preempt
- `ModelRunner`：组 batch、跑模型、采样
- `Engine.step()`：主循环
- 块级**哈希前缀缓存**（`ref_count` + 链式 hash）

不算子优化、不用 Radix、不混合 Prefill+Decode 于同一步。

## 文档

- [调度与 Continuous Batching](./docs/scheduling.md)

## 快速开始

```bash
cd vllm-v3
source ../vllm-v0/.venv/bin/activate

python run.py --model Qwen3-0.6B --temperature 0 --max-new-tokens 64
python run.py --model Qwen3-0.6B --batch --max-new-tokens 32

python test/test_block_manager.py
python test/test_scheduler.py
```

## 布局

```text
src/engine/scheduler.py    控制面
src/engine/model_runner.py 数据面组批
src/engine/engine.py       step / generate / generate_batch
src/memory/block_manager.py 分页 + 哈希前缀
src/memory/sequence.py     Sequence 状态机
```
