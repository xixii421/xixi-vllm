# Bench

吞吐、官方对照、硬件探测。结果写入 `out/bench.jsonl`。在仓库根目录跑。

| 脚本 | 做什么 |
| --- | --- |
| `bench_v4.py` | v4：单卡 / TP / PP / DP |
| `bench_vllm.py` | 官方 vLLM（只测单卡和 TP） |
| `run_matrix.py` | 按 suite 批量跑上面两个 |
| `report.py` | 把 jsonl 打成 Markdown 表 |
| `probe_server.py` | 拓扑、NCCL 延迟、这台机器适不适合 TP |
| `bench_bandwidth.py` | P2P / NCCL 带宽曲线 |

共享逻辑：`common.py`（jsonl）、`measure.py`（计时）、`cases.py`（矩阵用例）。

## 单配置

```bash
python bench/bench_v4.py --tp 1 --pp 1 --dp 1 --cg
python bench/bench_v4.py --tp 2 --cg
python bench/bench_v4.py --pp 2 --pp-layer-counts 12,28 --device-ids 1,0
python bench/bench_v4.py --dp 2 --n-reqs 32 --cg

python bench/bench_vllm.py --tp 1 --cg --cuda-devices 0
```

`dp=1`：TTFT、单请求 decode、batch-8。`dp>1`：N 路聚合吞吐。口径见 [docs/03-性能对比.md](../docs/03-性能对比.md)。

## 矩阵

```bash
python bench/run_matrix.py --suite smoke --reset-jsonl
python bench/run_matrix.py --suite full --reset-jsonl
python bench/run_matrix.py --suite vs-vllm --reset-jsonl --py-vllm /path/to/vllm/python
python bench/report.py
```

| suite | 内容 |
| --- | --- |
| `smoke` | 0.6B 单卡 / TP / PP，eager 与 cg |
| `full` | 对齐 docs/03 主表（本机两卡能跑的部分） |
| `vs-vllm` | 8B 上 v4 对官方 |

`run_matrix` 默认用当前 `python` 跑 v4。官方对照要另给 `--py-vllm` 或环境变量 `BENCH_PY_VLLM`。

## 硬件

```bash
python bench/probe_server.py --devices 0,1 --hidden 1024 --layers 28 --batch 1
python bench/bench_bandwidth.py --devices 0,1
python bench/bench_bandwidth.py --pairs PIX:0,1;PXB:2,3;SYS:0,2 --skip-nccl
```

说明见 [docs/02-服务器通信测量.md](../docs/02-服务器通信测量.md)。
