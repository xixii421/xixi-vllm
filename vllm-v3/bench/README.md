# vllm-v3 Benchmark

## 入口

```bash
cd vllm-v3 && source ../vllm-v0/.venv/bin/activate

python -m bench.run_all --suite decode --model Qwen3-1.7B
python -m bench.run_all --suite kv_vram --model Qwen3-1.7B --context-len 4096
python -m bench.run_all --suite all --model Qwen3-1.7B
```

## 输出

所有结果写入固定目录 ``bench/results/``（不再创建 ``{timestamp}_{model}/`` 子目录）：

| 文件 | 说明 |
| --- | --- |
| `meta.csv` | 最近一次运行元数据，含 `run_timestamp` |
| `report.md` | 汇总 Markdown；一次 `run_all` 多套件时追加章节 |
| `decode.csv` 等 | 各套件数据，运行该套件时覆盖 |

## 套件

与 v2 相同：`ttft` / `prefill` / `decode` / `e2e` / `vram` / `latency_seqlen` / `paged_overhead` / `kv_vram` / `paged`；`--suite all` 全跑。

v3 额外关注 **连续批处理** 下的 decode 吞吐与 v0–v2 对比，用仓库根目录：

```bash
cd ..   # repo root
source .venv/bin/activate && source .env.cuda.sh
python -m bench.compare --model Qwen3-0.6B --quick
python -m bench.compare --model Qwen3-0.6B --v3-compile
```
