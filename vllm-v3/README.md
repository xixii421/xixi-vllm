# vllm-v3 — 算子级性能提升

> 在 v2 三面架构上做系统性能优化：FlashAttention paged attention、Decode CUDA Graph、Triton KV 写、torch.compile prefill。

---

## 代码架构

```
vllm-v3/
├── run.py                        # 改写：+ --cg / --compile / --batch
├── config.yaml                   # 改写：+ kvcache_block_size / gpu_memory_utilization / torch_compile / enforce_eager
├── src/
│   ├── config.py                 # 改写：kvcache_block_size=256 / enforce_eager 等
│   ├── control/                  # 新增：控制面（v2 的 engine/ 拆出）
│   │   ├── engine.py             #   改写：step = schedule → run → postprocess
│   │   └── scheduler.py          #   改写：waiting/running、chunked prefill
│   ├── data/                     # 新增：数据面（v2 的 kv/ + 批数据拆出）
│   │   ├── batch.py              #   新增：常驻 GPU buffer + pin_memory H2D
│   │   ├── block_manager.py      #   新增：多请求 block 分配
│   │   ├── kv_ops.py             #   新增：Triton scatter 写核
│   │   └── manager.py            #   改写：KV 布局 [2,L,blocks,bs,n_kv,hd]
│   └── compute/                  # 新增：计算面（v2 的 layers/ + 推理编排拆出）
│       ├── cuda_graph.py         #   新增：Decode CUDA Graph
│       ├── executor.py           #   新增：执行入口
│       ├── model_runner.py       #   改写：decode/prefill 编排 + compile + CG fast path
│       ├── sampler.py            #   新增：批量采样
│       ├── layers/
│       │   ├── attention.py      #   改写：FlashAttention paged
│       │   └── ops.py            #   新增：FlashInfer RoPE/SiLU custom op
│       └── models/
│           ├── qwen3.py          #   改写：QKV / Gate-Up GEMM 融合
│           └── loader.py         #   新增：快速权重加载
├── bench/                        # 性能基准
├── test/                         # 测试
└── docs/                         # 设计文档
```

---

## 相对 v2 新增

| # | 优化 | 对应模块 | 收益 |
|:--:|------|---------|------|
| 1 | FlashAttention paged Attention | `src/compute/layers/attention.py` | 消除 gather + SDPA，直接 `block_table` paged 计算 |
| 2 | Decode 合批 | 同上 | `flash_attn_with_kvcache` 一次 kernel 处理整批 decode |
| 3 | Prefill with cache 消除 gather/mask | 同上 | `flash_attn_varlen_func(block_table)` varlen paged prefill，内核内因果 mask |
| 4 | Triton KV 写入 | `src/data/kv_ops.py` | fused scatter，替代 `index_copy_` |
| 5 | KV 布局对齐 paged attention | `src/data/manager.py` | `[2, L, blocks, block_size, n_kv, head_dim]`（NHD，block_size=256） |
| 6 | GQA 内核层处理 | FlashAttention 内核 | 不材料化 K/V，消除 `_expand_gqa()` |
| 7 | 上下文传递重构 | `src/compute/layers/attention.py` | `attention_context` 上下文管理器 + 模块级 `_ATTN_META`，CG 捕获兼容 |
| 8 | Decode CUDA Graph | `src/compute/cuda_graph.py` | `--cg`（`enforce_eager=False`）启用，decode 延迟大幅降低 |
| 9 | 快速权重加载 | `src/compute/models/loader.py` | `skip_param_init` + 目标 dtype 构建 + `load_file` 分片预取 |

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

# 启用 prefill torch.compile
python run.py --model Qwen3-0.6B --compile

# 批量推理
python run.py --model Qwen3-0.6B --batch

# 测试
PYTHONPATH=. pytest tests/test_scheduler.py -q
```

---

## 文档

| 文档 | 说明 |
|------|------|
| [00-优化总览.md](docs/00-优化总览.md) | v3 优化方向与三面划分 + 落地清单 |
| [01-v2性能瓶颈.md](docs/01-v2性能瓶颈.md) | v2 慢在哪（5 大瓶颈 A~E） |
| [02-实测证据.md](docs/02-实测证据.md) | 算子 → 机制 → 端到端 三层实测 |
| [03-输入管线.md](docs/03-输入管线.md) | Batch 输入管线 + pin_memory H2D |
| [04-显存计算.md](docs/04-显存计算.md) | 权重 / KV Cache 口算 + CG/compile 显存治理 |
| [05-算子融合.md](docs/05-算子融合.md) | GEMM 融合 + FlashInfer 融合核（RMSNorm / RoPE / SiLU，custom op 兼容 compile） |
| [06-Attention.md](docs/06-Attention.md) | FlashAttention paged attention（首 chunk / 续 chunk / decode） |
| [07-CUDA-Graph.md](docs/07-CUDA-Graph.md) | Decode CUDA Graph 原理 + v3 实现 |
| [08-Prefill-Compile.md](docs/08-Prefill-Compile.md) | Prefill torch.compile 全图编译 |
| [09-Sampler.md](docs/09-Sampler.md) | 批量采样 vs 逐样本 |
| [10-模型加载.md](docs/10-模型加载.md) | 快速权重加载 |
