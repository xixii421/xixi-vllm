# Bench 入口说明

主路径四个脚本，外加可选硬件探测。

| 脚本 | 作用 |
| --- | --- |
| bench_v4.py | v4：单卡、TP、PP、DP 及组合，支持 cg 与 compile |
| bench_vllm.py | 官方 vLLM 对照（仅单卡与 TP） |
| run_matrix.py | 按 suite 批量跑矩阵 |
| report.py | 从 out/bench.jsonl 生成 Markdown 表 |
| probe_server.py | 可选：服务器拓扑与带宽体检 |

组件微基准在 micro/ 目录，不进入日常矩阵。

## 单配置

在 vllm-v4 目录下执行。

单卡:

    ../.venv2/bin/python bench/bench_v4.py --tp 1 --pp 1 --dp 1 --cg

TP:

    ../.venv2/bin/python bench/bench_v4.py --tp 2 --pp 1 --dp 1 --cg

PP:

    ../.venv2/bin/python bench/bench_v4.py --tp 1 --pp 2 --dp 1 --cg

DP:

    ../.venv2/bin/python bench/bench_v4.py --tp 1 --pp 1 --dp 2 --n-reqs 32

官方对照:

    ../.venv-vllm/bin/python bench/bench_vllm.py --tp 1 --cg --cuda-devices 0

## 矩阵

    ../.venv2/bin/python bench/run_matrix.py --suite smoke --reset-jsonl
    ../.venv2/bin/python bench/run_matrix.py --suite full --reset-jsonl
    ../.venv2/bin/python bench/run_matrix.py --suite vs-vllm --reset-jsonl
    ../.venv2/bin/python bench/report.py

| suite | 内容 |
| --- | --- |
| smoke | 0.6B 单卡、TP、PP，eager 与 cg |
| full | 对齐 docs/02 的主矩阵 |
| vs-vllm | 8B 上 v4 与官方对照 |

结果统一写入 out/bench.jsonl。
