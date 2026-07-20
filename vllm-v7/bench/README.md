# Bench

两层，职责分离：

| 层 | 路径 | 用途 |
|----|------|------|
| **微基准** | `bench/lib` + `bench/suites` + `python -m bench.run` | 单卡 e2e / TTFT / prefill / decode / KV |
| **对比矩阵** | `scripts/bench_vs_vllm_full.py` | v7 vs 官方 vLLM（text / VL / MoE × TP/PP × CG） |

## 微基准（推荐日常）

核心库在 `bench/lib/`（`BenchConfig` / `BenchSession` / timing / report）。

```bash
cd vllm-v7
PYTHONPATH=. python -m bench.run --suite e2e,ttft,decode
PYTHONPATH=. python -m bench.run --suite all
```

套件：`e2e` `ttft` `prefill` `decode` `kv_vram`。结果写入 `bench/results/`。

## 官方对比

```bash
PYTHONPATH=. python scripts/bench_vs_vllm_full.py --mode cg --suite text,vl,moe
```

需 `.venv`（v7）与 `.venv-vllm`（官方）。分析报告见 `bench/results/PERF_ANALYSIS_*.md`。
