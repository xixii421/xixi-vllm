# v3 bench

按「算子 → 正确性 → 机制 → 端到端 → 跨引擎」顺序跑。结果只写 `bench/out/`（每次全套重跑会先清空再写真实数据）。

在 `vllm-v3/` 下：

```bash
# 全套（推荐）：清空 out/ 后跑 01-07；官方 vLLM 走 .venv-vllm
../.venv2/bin/python bench/run_all.py

# 或逐个
../.venv2/bin/python bench/01_ops.py
../.venv2/bin/python bench/02_correctness.py
../.venv2/bin/python bench/03_prefill.py
../.venv2/bin/python bench/04_prefix.py
../.venv2/bin/python bench/05_scheduler.py
../.venv2/bin/python bench/06_e2e.py
../.venv2/bin/python bench/07_compare.py --engine v3
../.venv2/bin/python bench/07_compare.py --engine nano
../.venv-vllm/bin/python bench/07_compare.py --engine vllm
```

| 顺序 | 脚本 | 回答什么 | 输出 |
|:---:|------|----------|------|
| 01 | `01_ops.py` | 每个被改的算子相对 v2 快多少（CUDA event 中位数） | `out/01_ops.md` |
| 02 | `02_correctness.py` | greedy 可复现；compile 能跑完（不要求 bit 一致） | `out/02_correctness.md` |
| 03 | `03_prefill.py` | 短/长 prompt 的稳态 TTFT：eager vs compile | `out/03_prefill.md` |
| 04 | `04_prefix.py` | none / hash / radix 命中后的 TTFT | `out/04_prefix.md` |
| 05 | `05_scheduler.py` | mix 开/关的 dec@P、short_P | `out/05_scheduler.md` |
| 06 | `06_e2e.py` | eager / decode-CG / CG+compile 的连续批吞吐 | `out/06_e2e.md` |
| 07 | `07_compare.py` | 同一 workload 对比 v3、nano-vllm、官方 vLLM | `out/07_compare.md` |

共享逻辑在 `common.py`：引擎工厂、等长 dummy、**3 轮 greedy warmup**（compile → 录图 → 重放），计时不含 tokenizer/print。

口径：

- TTFT = 第一个 completion token；greedy；warmup 之后取中位数
- Decode / 连续批吞吐见 `06_e2e.py`、`07_compare.py`
- `03_prefill` 两边都开 decode CG，只隔离 `torch_compile`
- `07` 因 nano 禁止 greedy，三边统一 `temperature=0.6`、`ignore_eos`；16 条请求，prompt 100~400，out 50~150
- 三个引擎不能同进程混 import，07 按 `--engine` 分三次跑
