# 待完善

代码量和知识量太多，等待完善中......

---

## 单卡性能基准（2026-08-16）

与 `vllm-v3` 同口径实测（`bench/06_e2e.py`，16 条变长请求 prompt 100~400 / out 32~64，greedy + ignore_eos，
4090 D，Qwen3-0.6B，.venv2 = PyTorch 2.13 / FlashInfer 0.6.15 / Triton 3.7.1）：

| 档位 | v5 (tok/s, 中位) | v3 (tok/s, 中位) | v5/v3 |
| --- | ---: | ---: | ---: |
| eager | 752.4 | 776.0 | 0.97x |
| cg（decode CUDA Graph） | 2122.0 | 2054.4 | 1.03x |

两版单卡性能在 ±3% 内，差异属工程设计取舍而非性能代差。完整结论见 `vllm-v3/docs/02-实测证据.md` 3.7 节。
