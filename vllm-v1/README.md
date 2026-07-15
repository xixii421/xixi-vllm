# vllm-v1 — KV Cache

在 v0 朴素自回归推理的基础上，v1 引入 **连续 KV Cache**，把 prefill 与 decode 拆开，decode 每步只算 1 个新 token，复杂度从 O(n²) 降到 O(n)。

## 与 v0 的差异

| 模块 | v0 | v1 |
| --- | --- | --- |
| `engine.py` | 每步重算整段序列 | prefill 一次 + decode 循环 |
| `layers/attention.py` | 纯 SDPA causal | 写 cache + prefill/decode 分支 |
| `kv_cache.py` | 无 | 每层 `[max_len, kv_heads, head_dim]` |
| `utils/context.py` | 无 | thread-local `ForwardContext` |

## 快速开始

```bash
cd vllm-v1
source ../vllm-v0/.venv/bin/activate   # 或 ./setup_env.sh
python run.py --model Qwen3-0.6B --prompt "Hello" --max-new-tokens 64 --temperature 0
```

## Benchmark

```bash
python bench/bench_decode.py
python bench/bench_latency_seqlen.py
```

v1 预期表现：
- **Prefill**：与 v0 同量级，随 seq_len 略增
- **Decode**：latency 基本平坦（~常数），tok/s 显著高于 v0

## 设计文档

详见 [docs/kv_cache.md](docs/kv_cache.md)。
