# vllm-v0 Benchmark

`bench/` 根目录只保留统一入口 `run_all.py`，实现代码在 `lib/` 与 `suites/` 中。

## 目录结构

```
bench/
  run_all.py          # 唯一入口
  README.md
  results/            # 输出目录
  lib/                # 配置、计时、session、报告
    config.py
    cli.py
    timing.py
    session.py
    report.py
    registry.py
  suites/             # 各指标测量逻辑
    ttft.py
    prefill.py
    decode.py
    e2e.py
    vram.py
    latency_seqlen.py
```

## 运行

```bash
cd vllm-v0
source .venv/bin/activate

# 全部指标
python -m bench.run_all

# 单个套件
python -m bench.run_all --suite decode
python -m bench.run_all --suite latency_seqlen --lengths 128 512 1024 2048

# 自定义参数
python -m bench.run_all --model Qwen3-0.6B --warmup 5 --iters 10 --context-len 4096
```

## v0 vs v1 对比

在仓库根目录：

```bash
python scripts/compare_v0_v1_bench.py --quick
```

每次运行写入固定目录 ``bench/results/``（覆盖同名校准文件，不再按时间戳分子目录）：

```
bench/results/
  meta.csv       # 最近一次运行的元数据（含 run_timestamp）
  report.md      # 汇总报告（一次 run_all 内多套件会追加章节）
  decode.csv
  ttft.csv
  ...
```
