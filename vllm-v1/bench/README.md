# vllm-v1 Benchmark

在 v0 结构基础上，`session.py` 感知 KV Cache；`latency_seqlen` 区分 **prefill** 与 **decode_step**。

## 目录结构

```
bench/
  run_all.py          # 唯一入口
  README.md
  results/
  lib/
  suites/
```

## 运行

```bash
cd vllm-v1
source ../vllm-v0/.venv/bin/activate

python -m bench.run_all
python -m bench.run_all --suite decode
python -m bench.run_all --suite latency_seqlen --lengths 128 512 1024 2048
```

## v0 vs v1 对比

```bash
cd ..   # 仓库根目录
python scripts/compare_v0_v1_bench.py --quick
```

v1 预期：Decode tok/s 显著高于 v0；Decode step latency 随上下文长度基本平坦。

## 输出

结果固定写入 ``bench/results/``（``meta.csv``、``report.md``、各套件 ``*.csv``），每次运行覆盖同名文件；时间戳记录在 ``meta.csv`` 的 ``run_timestamp`` 字段。

设计说明见 [docs/kv_cache.md](../docs/kv_cache.md)。
