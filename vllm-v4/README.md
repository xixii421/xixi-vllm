# vllm-v4

主题：**算子优化 + Continuous Batching**（FlashInfer / CUDA Graph / Triton KV / Triton Norm+RoPE）。

在 v3 调度架构上替换数据面：

- **FlashInfer** paged prefill/decode attention
- **Triton** `store_kvcache` + **fused RMSNorm** + **RoPE**
- **torch.compile** `SiluAndMul`
- **动态 KV 池**（按 `gpu_memory_utilization` 自动 sizing，`block_size=256`）
- **Decode CUDA Graph**（`--compile` 启用）
- **pin_memory 输入管线** + **批量采样**
- 保留 v3：`Scheduler`、哈希前缀 `BlockManager`、`Engine.step`


## 文档

- [调度与 Continuous Batching](./docs/scheduling.md)
- [算子优化（FlashInfer / CG / Triton）](./docs/operators.md)
- [优化全景与 FlashInfer CG 路线](./docs/optimization_guide.md)

## 快速开始

```bash
cd vllm-v4
bash setup_env.sh
source .venv/bin/activate

python run.py --model Qwen3-0.6B --temperature 0 --max-new-tokens 64
python run.py --model Qwen3-0.6B --compile          # 启用 Decode CUDA Graph
python run.py --model Qwen3-0.6B --batch --max-new-tokens 32

python test/test_block_manager.py
python test/test_scheduler.py
```

## 布局

```text
src/layers/attention_backend.py  FlashInfer prefill/decode
src/layers/attention.py          Triton KV + wrapper.run
src/memory/kv_ops.py             Triton store_kvcache
src/core/cuda_graph.py           Decode CUDA Graph
src/engine/model_runner.py       动态 KV + warmup + CG fast path
src/engine/input_builder.py      pin_memory 输入
src/engine/sampler.py            批量采样
src/engine/scheduler.py          控制面（同 v3）
```

## v3 → v4 差异

| 能力 | v3 | v4 |
| --- | --- | --- |
| Attention | PyTorch gather+matmul | FlashInfer paged |
| KV 写入 | `index_copy_` | Triton scatter |
| block_size | 16 | **256** |
| KV 池 | 固定公式 | **动态 VRAM sizing** |
| Decode | eager only | **CUDA Graph**（可选） |
| Prefill batch | 1D varlen | 1D varlen + FlashInfer |
| SiluAndMul | eager | **torch.compile** |
| RMSNorm | PyTorch | **Triton fused** |
| RoPE | PyTorch | **Triton** |
| 采样 | per-seq | **batched Sampler** |
