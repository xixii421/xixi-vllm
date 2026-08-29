# vllm-v5 — MoE（稀疏专家）

> 在 v4 稠密并行引擎上加入 Mixture of Experts：稀疏门控路由、grouped GEMM 融合、专家并行（EP 复用 TP 组）、MoE 进 Decode CUDA Graph。

---

## 代码架构

```
vllm-v5/
├── run.py                        # 改写：+ --ep / --moe-backend / --eager / --no-compile
├── config.yaml                   # 改写：+ enable_ep / moe_backend
├── src/
│   ├── config.py                 # 改写：+ enable_ep / moe_backend / is_moe
│   ├── moe/                      # 新增：稀疏专家
│   │   ├── block.py              #   SparseMoeBlock：gate → route → experts
│   │   ├── experts.py            #   3D 权重、expert_map、HF 加载
│   │   ├── parallel.py           #   MoEParallelConfig（TP / EP）
│   │   ├── fused.py              #   fused_moe_forward 后端分发
│   │   └── kernels/
│   │       ├── triton.py         #   grouped GEMM
│   │       ├── cuda.py           #   JIT 加载 + workspace 池化
│   │       └── moe_kernels.cu    #   align / mul_and_silu / moe_sum
│   ├── compute/
│   │   ├── worker.py             # 改写：resolve_moe_backend + MoEParallelConfig
│   │   ├── layers/
│   │   │   └── linear.py         # 改写：+ ReplicatedLinear（共享专家）
│   │   └── models/
│   │       ├── qwen3_moe.py      # 新增：Qwen3MoeForCausalLM
│   │       └── __init__.py       # 改写：注册 qwen3_moe
│   ├── distributed/
│   │   └── parallel_state.py     # 改写：EP 复用 TP 组
│   └── utils/                    # 新增：工具
│       ├── profiler.py           #   性能剖析
│       ├── cuda_env.py           #   CUDA 环境检测
│       └── paths.py              #   路径解析
├── tests/                        # 新增：MoE 单测 / GPU / 分布式
├── bench/                        # MoE 基准（moe_e2e / vs_vllm）
└── docs/                         # MoE 文档
```

---

## 相对 v4 新增

| 新增能力 | 对应模块 | 说明 |
|---------|---------|------|
| 稀疏专家层 | `src/moe/block.py` | SparseMoeBlock：gate → route_experts → experts → shared/AR |
| 专家权重与加载 | `src/moe/experts.py` | 3D `w13`/`w2`、`expert_map`、HF 两种存法 |
| 专家并行 EP | `src/moe/parallel.py` | MoEParallelConfig，复用 TP 组切 expert，token 不搬家 |
| fused MoE 后端 | `src/moe/fused.py` + `kernels/` | cutlass / triton grouped GEMM + align/silu/sum 核 |
| MoE 模型 | `src/compute/models/qwen3_moe.py` | Qwen3MoeForCausalLM，共享专家 + routed 专家 |
| MoE 进 CUDA Graph | `src/compute/worker.py` / `kernels/cuda.py` | workspace 按容量池化，decode 整图 capture |

---

## 单卡性能基准（2026-08-16）

与 `vllm-v3` 同口径实测（`bench/06_e2e.py`，16 条变长请求 prompt 100~400 / out 32~64，greedy + ignore_eos，
4090 D，Qwen3-0.6B，.venv2 = PyTorch 2.13 / FlashInfer 0.6.15 / Triton 3.7.1）：

| 档位 | v5 (tok/s, 中位) | v3 (tok/s, 中位) | v5/v3 |
| --- | ---: | ---: | ---: |
| eager | 752.4 | 776.0 | 0.97x |
| cg（decode CUDA Graph） | 2122.0 | 2054.4 | 1.03x |

两版单卡性能在 ±3% 内，差异属工程设计取舍而非性能代差。完整结论见 `vllm-v3/docs/02-实测证据.md` 3.7 节。

---

## 文档

| 文档 | 说明 |
|------|------|
| [00-MoE概述.md](docs/00-MoE概述.md) | 从 Jacobs 到 Qwen3-MoE：稀疏门控、Switch、Mixtral、DeepSeek |
| [01-路由与融合计算.md](docs/01-路由与融合计算.md) | naive 双重循环 → align / grouped GEMM → v5 前向 |
| [02-专家并行.md](docs/02-专家并行.md) | MoE TP 与 EP、v5 不搬 token、共享专家与 AR 顺序 |
| [03-v5实现.md](docs/03-v5实现.md) | `src/moe/` 文件地图、权重加载、后端、与官方差异 |
