# vllm-v3 — 算子级性能提升

> 在 v2 三面架构上做系统性能优化：FlashInfer paged attention、CUDA Graph decode、Triton 算子。

---

## 相对 v2 新增（8 大优化）

| # | 优化 | 对应模块 | 收益 |
|:--:|------|---------|------|
| 1 | FlashInfer paged Attention | `src/layers/attention_backend.py` | 消除 gather + SDPA，直接 paged KV 计算 |
| 2 | Decode 合批 | 同上 | 一次 kernel 处理整批 decode，不再逐样本 SDPA |
| 3 | Prefill with cache 消除 gather/mask | 同上 | varlen paged prefill，内核内因果 mask |
| 4 | Triton KV 写入 | `src/memory/kv_ops.py` | fused scatter，替代 `index_copy_` |
| 5 | KV 布局对齐 FlashInfer | `src/engine/model_runner.py` | NHD 布局，block_size=256 |
| 6 | GQA 内核层处理 | FlashInfer plan | 不材料化 K/V，消除 `_expand_gqa()` |
| 7 | contextvars 上下文传递 | `src/layers/attention.py` | 线程安全，CG 兼容 |
| 8 | Decode CUDA Graph | `src/core/cuda_graph.py` | `--compile` 启用，decode 延迟大幅降低 |

**结果：** decode 吞吐从 v2 的 ~40 tok/s 提升到 ~400 tok/s（Qwen3-0.6B）。

---

## 快速开始

```bash
cd vllm-v3

# 单条推理（eager 模式）
python run.py --model Qwen3-0.6B --temperature 0

# 启用 Decode CUDA Graph
python run.py --model Qwen3-0.6B --compile

# 批量推理
python run.py --model Qwen3-0.6B --batch

# 测试
PYTHONPATH=. pytest tests/test_scheduler.py -q
```

---

## 核心源码

| 文件 | v2→v3 |
|------|-------|
| `src/layers/attention_backend.py` | **新增** — FlashInferBackend + build_metadata + CSR 索引构建 |
| `src/layers/attention.py` | **改写** — `store_kvcache → wrapper.run`，contextvars 上下文 |
| `src/memory/kv_ops.py` | **新增** — Triton `store_kvcache` kernel |
| `src/core/cuda_graph.py` | **新增** — CUDAGraphRunner decode |
| `src/engine/model_runner.py` | **改写** — 动态 KV pool sizing + warmup + CG fast path |
| `src/engine/input_builder.py` | **改写** — pin_memory H2D |
| `src/engine/sampler.py` | **新增** — 批量采样 |
| `src/config.py` | **扩展** — `kvcache_block_size=256` / `gpu_memory_utilization` / FlashInfer 开关 |

---

## 文档

| 文档 | 说明 |
|------|------|
| [01-v2性能瓶颈.md](docs/01-v2性能瓶颈.md) | v2 慢在哪（3 大瓶颈） |
| [07-Attention优化.md](docs/07-Attention优化.md) | 8 大优化详解 + FlashInfer vs FlashAttention |
| [03-实测证据.md](docs/03-实测证据.md) | bench 数据（吞吐、延迟、前缀命中） |
| [04-CUDA-Graph.md](docs/04-CUDA-Graph.md) | CUDA Graph 原理 + v3 实现 |
| [05-Input.md](docs/05-Input.md) | pin_memory + H2D 优化 |
| [06-Sampler.md](docs/06-Sampler.md) | 批量采样 vs 逐样本 |
| [08-RMSNorm.md](docs/08-RMSNorm.md) | Triton RMSNorm 性能实测 |
