# Qwen3-0.6B 推理 Benchmark：朴素推理 / v3 / vLLM

测试日期：2026-09-17

## 结论

在 RTX 4060 Laptop GPU 上，固定 8 个请求、每请求输入 256 token 并输出 64 token：

- 朴素推理（v0，全上下文重算）：19.51 tok/s
- 本项目 v3：886.92 tok/s，是朴素推理的 **45.46x**
- 官方 vLLM 0.29.0：854.35 tok/s，是朴素推理的 **43.79x**
- 本项目 v3 比同机 vLLM 高 **3.81%**

单请求时，v3 为 149.21 tok/s，是朴素推理的 7.04x，比 vLLM 的 140.00 tok/s 高 6.58%。

## 环境

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB，driver 610.88
- CPU：Intel Core i7-13650HX
- 系统：Windows 11 / WSL2，Linux 6.18.33.2
- 模型：Qwen3-0.6B，BF16
- 公共运行栈：Python 3.12.13，PyTorch 2.13.0+cu130，FlashAttention 2.8.3
- 对照版本：官方 vLLM 0.29.0，Transformers 5.17.0

## 公平性口径

- 三方使用完全相同的 deterministic token IDs，不经过 tokenizer。
- 每请求输入 256 token，固定生成 64 token，greedy，并忽略 EOS。
- 计时不含模型加载、tokenizer、首次 JIT/compile、CUDA Graph 捕获和输出打印。
- 每个引擎先运行短 warmup，再执行一次与正式 workload 完全同形的完整 warmup。
- 正式 workload 连续运行 5 次，主结果取吞吐中位数；每轮均调用 `torch.cuda.synchronize()`。
- 关闭 prefix cache，避免重复测量被前缀命中加速。

三种引擎的含义：

- **朴素推理 v0**：没有 KV Cache；每生成一个 token，都对不断增长的完整上下文重新执行 forward。多个请求只能顺序处理。
- **本项目 v3**：Paged KV Cache、Continuous Batching、Paged FlashAttention、Decode CUDA Graph 和 `torch.compile`。
- **官方 vLLM 0.29.0**：离线 `LLM.generate`，FlashAttention 2、Continuous Batching、`torch.compile` 和 CUDA Graph。

## 单请求结果

每轮生成 64 token。

| 引擎 | 五轮吞吐（tok/s） | 中位数（tok/s） | 相对朴素推理 | 相对 vLLM |
| --- | --- | ---: | ---: | ---: |
| 朴素推理 v0 | 21.19 / 21.20 / 21.74 / 20.72 / 20.94 | 21.19 | 1.00x | 0.15x |
| 本项目 v3 | 148.56 / 149.44 / 147.64 / 149.26 / 149.21 | **149.21** | **7.04x** | **1.066x** |
| vLLM 0.29.0 | 138.54 / 140.00 / 140.57 / 139.82 / 140.65 | 140.00 | 6.61x | 1.00x |

## 8 请求吞吐结果

三方处理同一组 8 个请求，共生成 512 token。朴素 v0 顺序处理；v3 和 vLLM 使用连续批处理。

| 引擎 | 五轮吞吐（tok/s） | 中位数（tok/s） | 中位 wall time（s） | 相对朴素推理 | 相对 vLLM |
| --- | --- | ---: | ---: | ---: | ---: |
| 朴素推理 v0 | 22.62 / 21.51 / 15.07 / 19.32 / 19.51 | 19.51 | 26.24 | 1.00x | 0.023x |
| 本项目 v3 | 878.72 / 886.92 / 888.91 / 882.71 / 890.77 | **886.92** | **0.577** | **45.46x** | **1.038x** |
| vLLM 0.29.0 | 862.06 / 846.68 / 854.35 / 860.83 / 848.73 | 854.35 | 0.599 | 43.79x | 1.00x |

v0 的长时间连续全量重算更容易受到 Laptop GPU 温度/功耗波动影响，因此五轮范围较宽；报告采用预先约定的中位数。v3 和 vLLM 的五轮范围分别为 878.72～890.77 与 846.68～862.06 tok/s。

## vLLM 在 WSL2 下的配置说明

vLLM 0.29.0 默认的 V2 Model Runner 需要 UVA；本机 WSL2 被 vLLM 检测为不支持 pinned memory，默认路径会报 `UVA is not available`。因此实测采用 vLLM 官方保留的兼容配置：

- `VLLM_USE_V2_MODEL_RUNNER=0`
- `VLLM_ENABLE_V1_MULTIPROCESSING=0`
- `attention_backend="FLASH_ATTN"`
- `VLLM_USE_FLASHINFER_SAMPLER=0`（本测试为 greedy，不需要 top-k/top-p sampler）
- `enable_prefix_caching=False`
- `gpu_memory_utilization=0.85`
- `max_model_len=4096`

这仍是官方 vLLM 0.29.0 的 V1 engine 和 GPU runner；主 attention 后端为 FlashAttention 2。该限制意味着结果代表“本机 Win11/WSL2 可运行配置”，不代表原生 Linux 上默认 V2 Model Runner 的性能。

初始化时 vLLM 对与 Qwen3 无关的可选 Numba/TileLang warmup 给出缺少 `llvmlite`/`z3` 的 warning，但 Qwen3 模型加载、CUDA Graph 捕获、完整 warmup 和全部 5 轮生成均成功，输出 token 数全部匹配目标。

## v3 自身执行模式消融（补充）

8 个变长请求；输入 100～400 token，输出 32～64 token；三次独立进程运行取中位数。

| 模式 | 三轮吞吐（tok/s） | 中位数（tok/s） | 相对 eager |
| --- | --- | ---: | ---: |
| eager | 300.1 / 252.7 / 288.0 | 288.0 | 1.00x |
| Decode CUDA Graph | 574.4 / 635.1 / 652.4 | 635.1 | 2.21x |
| CUDA Graph + `torch.compile` | 684.2 / 612.5 / 640.3 | 640.3 | 2.22x |

CUDA Graph 是该 GPU 上主要且稳定的收益。完整模式较 eager 提升 122.3%，但 `torch.compile` 相对单独 CUDA Graph 的中位数仅提升 0.8%。

## Prefix Cache（补充）

Radix Prefix Cache 测试使用 1024-token 公共前缀和 128-token 独特后缀；每项 8 次取中位数：

- miss TTFT：54.38 ms
- 命中 1024 token 后 TTFT：27.80 ms
- TTFT 缩短 48.9%（1.96x）

## 简历建议表述

推荐版本：

> **测试与评测：** 构建统一 token-level benchmark；在 RTX 4060 Laptop 上实测 Qwen3-0.6B（BF16，256-token 输入/64-token 输出，5 次中位数），v3 将 8 请求吞吐从朴素全上下文重算的 19.5 提升至 886.9 tok/s（45.5x），并较同机官方 vLLM 0.29.0 的 854.4 tok/s 高 3.8%。

更短版本：

> **性能优化：** 在 RTX 4060 Laptop 上，Paged KV Cache、Continuous Batching 与 Decode CUDA Graph 将 8 请求吞吐从朴素推理的 19.5 提升至 886.9 tok/s（45.5x），较同机 vLLM 0.29.0 高 3.8%。

## 复现

三方脚本：`vllm-v3/bench/bench_three_way.py`

本地 v0/v3：

```bash
.venv/bin/python vllm-v3/bench/bench_three_way.py \
  --engine v0 --num-seqs 8 --prompt-len 256 --output-len 64 --repeats 5

.venv/bin/python vllm-v3/bench/bench_three_way.py \
  --engine v3 --num-seqs 8 --prompt-len 256 --output-len 64 --repeats 5
```

官方 vLLM 需要单独安装 0.29.0，并在本机 WSL2 使用：

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_USE_V2_MODEL_RUNNER=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
python vllm-v3/bench/bench_three_way.py \
  --engine vllm --num-seqs 8 --prompt-len 256 --output-len 64 --repeats 5
```

