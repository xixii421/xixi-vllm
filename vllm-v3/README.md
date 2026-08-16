# vllm-v3 — 算子级性能提升

> 在 v2 三面架构上做系统性能优化：FlashInfer paged attention、CUDA Graph decode、Triton 算子。

---

## 相对 v2 新增

| # | 优化 | 对应模块 | 收益 |
|:--:|------|---------|------|
| 1 | FlashInfer paged Attention | `src/layers/attention_backend.py` | 消除 gather + SDPA，直接 paged KV 计算 |
| 2 | Decode 合批 | 同上 | 一次 kernel 处理整批 decode，不再逐样本 SDPA |
| 3 | Prefill with cache 消除 gather/mask | 同上 | varlen paged prefill，内核内因果 mask |
| 4 | Triton KV 写入 | `src/memory/kv_ops.py` | fused scatter，替代 `index_copy_` |
| 5 | KV 布局对齐 FlashInfer | `src/engine/model_runner.py` | NHD 布局，block_size=256 |
| 6 | GQA 内核层处理 | FlashInfer plan | 不材料化 K/V，消除 `_expand_gqa()` |
| 7 | contextvars 上下文传递 | `src/layers/attention.py` | 线程安全，CG 兼容 |
| 8 | Decode CUDA Graph | `src/core/cuda_graph.py` | `--cg` 启用，decode 延迟大幅降低 |
| 9 | 快速权重加载 | `src/models/loader.py` | `skip_param_init` + 目标 dtype 构建 + `load_file` 分片预取 |

**结果：** decode 吞吐从 v2 的 ~40 tok/s 提升到 ~400 tok/s（Qwen3-0.6B）。
**加载：** 4B 模型权重加载从 ~29s 降到 ~3.9s（跳过随机初始化 + 整分片预取）。

---

## 快速开始

```bash
cd vllm-v3

# 单条推理（eager 模式）
python run.py --model Qwen3-0.6B --temperature 0

# 启用 Decode CUDA Graph
python run.py --model Qwen3-0.6B --cg

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
| [00-优化总览.md](docs/00-优化总览.md) | v3 优化方向与三面划分 |
| [01-v2性能瓶颈.md](docs/01-v2性能瓶颈.md) | v2 慢在哪（5 大瓶颈 A~E） |
| [02-实测证据.md](docs/02-实测证据.md) | 算子 → 机制 → 端到端 三层实测（含 piece CG/前缀命中/dec@P/short_P/多引擎对比，附完整测试代码） |
| [04-Attention.md](docs/04-Attention.md) | FlashInfer paged attention + FlashInfer vs FlashAttention |
| [05-算子融合.md](docs/05-算子融合.md) | Triton RMSNorm / RoPE / SiLU 融合 |
| [06-CUDA-Graph.md](docs/06-CUDA-Graph.md) | Decode CUDA Graph 原理 + v3 实现 |
| [07-Prefill-CUDA-Graph.md](docs/07-Prefill-CUDA-Graph.md) | Prefill piecewise CUDA Graph |
| [08-显存优化.md](docs/08-显存优化.md) | CG workspace / graph pool 显存管理 |
| [09-输入管线.md](docs/09-输入管线.md) | pin_memory + H2D 优化 |
| [10-Sampler.md](docs/10-Sampler.md) | 批量采样 vs 逐样本 |
| [11-模型加载.md](docs/11-模型加载.md) | 快速权重加载 |
