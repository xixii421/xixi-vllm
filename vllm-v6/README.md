# vllm-v6

主题：**Megatron 风格 TP + PP** — 在 v5（FlashInfer + Continuous Batching + Qwen3-VL）之上接入张量并行 / 流水线并行，用 `multiprocessing.spawn` + NCCL（不用 torchrun）。

## 快速开始

```bash
cd vllm-v6
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.
source .venv/bin/activate   # symlink → vllm-v5/.venv

# 单卡（与 v5 行为一致）
python run.py --model ~/huggingface/Qwen3-0.6B --temperature 0 --max-new-tokens 64 --eager

# Tensor Parallel = 2（两张卡）
python run.py --model ~/huggingface/Qwen3-0.6B \
  --tp-size 2 --device-ids 0,1 --temperature 0 --max-new-tokens 64 --eager

# Pipeline Parallel = 2
python run.py --model ~/huggingface/Qwen3-0.6B \
  --pp-size 2 --device-ids 0,1 --temperature 0 --max-new-tokens 64 --eager
```

`--device-ids` 按 **global rank** 映射物理 GPU（例如 `0,1` = 4070 + 4090）。

## 架构增量（相对 v5）

```text
src/distributed/          WORLD / TP / PP groups + NCCL ops
src/layers/linear.py      Column / Row / QKV / Merged parallel
src/layers/vocab_parallel_embedding.py
src/layers/utils_parallel.py   IntermediateTensors, PPMissingLayer
src/models/qwen3.py       make_layers + parallel linears + weight loaders
src/engine/worker.py      per-rank init + PP recv/send
src/engine/executor.py    spawn tp×pp；Pipe RPC；driver = last PP & tp_rank0
```

- `world_size == 1`：进程内 `ModelRunner`（仍 `initialize_model_parallel(1,1)`）
- `world_size > 1`：`Executor` 拉起多进程；调度仍在主进程
- 多模态（Qwen3-VL）：支持 TP/PP；ViT 按 `mm_encoder_tp_mode` 切分：
  - `data`（默认）：各 TP 保留完整 ViT，按 patch 负载切 batch，再 all-gather 图像嵌入
  - `weights`：ViT 内部 Column/Row/QKV 权重并行
- Decode CUDA Graph：按 PP stage 本地捕获（`--compile`）；PP 通信在图外

## 测试

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.
./.venv/bin/python -m pytest -m unit -q
./.venv/bin/python -m pytest -m "unit or gpu" -q
./.venv/bin/python -m pytest -m distributed -q --tb=short
```

## 性能矩阵（TP/PP × CG，文本 / 多模态分开跑）

```bash
# 文本：single / TP=2 / PP=2 × eager / CG
python bench/bench_parallel_matrix.py --domain text \
  --model ~/huggingface/Qwen3-0.6B \
  --modes single,tp2,pp2 --cg-modes eager,cg --device-ids 0,1

# 多模态：同样矩阵；VL+PP 建议 first stage 放 4090（device-ids 1,0）
python bench/bench_parallel_matrix.py --domain vl \
  --model ~/huggingface/Qwen3-VL-2B-Instruct \
  --modes single,tp2,pp2 --cg-modes eager,cg --device-ids 1,0
```

输出：`bench/results/parallel_{text|vl}.{json,md}`（TTFT / prefill / decode / e2e / VRAM）。

详见 `tests/README.md` 与 `DESIGN.md`。

## 相对 v5 的不做项

- 多机 / IB
- torchrun 启动（本版刻意用 spawn 教学）
