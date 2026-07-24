# vllm-v3

主题：**在 v2 三面架构上做算子级性能提升**（FlashInfer / CUDA Graph / Triton）。

控制面 / 资源层与 v2 同构（`Engine` → `Scheduler(kv)` → `KVManager` → `ModelRunner.run(prefills, decodes)`）；数据面换成高性能路径。

## 相对 v2 新增 / 替换的模块

| 模块 | 路径 | 作用 |
| --- | --- | --- |
| FlashInfer Attention | `src/layers/attention_backend.py` | paged prefill / decode，替代 gather+SDPA |
| Triton KV 写入 | `src/memory/kv_ops.py` | `store_kvcache` scatter，替代 `index_copy_` |
| Triton Norm / RoPE | `src/layers/layernorm.py`, `rotary.py` | fused RMSNorm + RoPE |
| torch.compile 激活 | `src/layers/activation.py` | `SiluAndMul` |
| Decode CUDA Graph | `src/core/cuda_graph.py` | 可选 `--compile` |
| pin_memory 输入 | `src/engine/input_builder.py` | H2D 异步拷贝 |
| 批量采样 | `src/engine/sampler.py` | 一次处理整 batch |
| 动态 KV 池 | `ModelRunner` + `EngineConfig` | 按 `gpu_memory_utilization` sizing，`block_size=256` |
| PD 混合调度 | `src/engine/scheduler.py` | decode-first + 剩余 budget 打一块 prefill（默认开） |

## 文档

- [相比 v2 的增改](./docs/01-相比v2的增改.md)
- [调度与 Continuous Batching](./docs/scheduling.md)
- [算子优化（FlashInfer / CG / Triton）](./docs/operators.md)
- [优化全景与 FlashInfer CG 路线](./docs/optimization_guide.md)

## 快速开始

```bash
cd vllm-v3
bash setup_env.sh
source .venv/bin/activate

python run.py --model Qwen3-0.6B --temperature 0 --max-new-tokens 64
python run.py --model Qwen3-0.6B --compile          # Decode CUDA Graph
python run.py --model Qwen3-0.6B --batch --max-new-tokens 32

PYTHONPATH=. pytest tests/test_scheduler.py -q
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
src/engine/scheduler.py          CB + chunked prefill + PD mix
```

## v2 → v3 差异（数据面）

| 能力 | v2 | v3 |
| --- | --- | --- |
| Attention | PyTorch gather + SDPA | FlashInfer paged |
| KV 写入 | `index_copy_` / gather | Triton scatter |
| block_size | 16（教学） | **256**（FlashInfer） |
| KV 池 | 固定 / 显式 | **动态 VRAM sizing** |
| Decode | eager only | **CUDA Graph**（可选） |
| RMSNorm / RoPE | PyTorch | **Triton** |
| SiluAndMul | eager | **torch.compile** |
| 采样 | per-seq | **batched Sampler** |
| 调度 | CB + PD mix | 同左（默认 mix） |
