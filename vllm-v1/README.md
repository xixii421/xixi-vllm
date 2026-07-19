# vllm-v1 — KV Cache + 前缀索引

在 v0 朴素自回归之上，v1 引入 **block 池 KV** 与可选的 **hash / radix 前缀缓存**。  
非 KV 代码与 [vllm-v0](../vllm-v0/) 对齐；KV 全部在 `src/kv/`。

相对 v0 的文件清单见 [docs/ADDED_vs_v0.md](docs/ADDED_vs_v0.md)。

## 模块一览

| 路径 | 相对 v0 | 作用 |
| --- | --- | --- |
| `src/kv/` | **新增** | KV 专模块 |
| `src/kv/pool.py` | **新增** | 物理 block 池 |
| `src/kv/prefix/hash_index.py` | **新增** | 链式 block-hash 索引 |
| `src/kv/prefix/radix_*.py` | **新增** | 基数树前缀索引 |
| `src/engine.py` | **改写** | prefill + decode + `KVManager` |
| `src/layers/attention.py` | **改写** | write/gather + SDPA |
| 其余 `src/*` | 同步 v0 | 模型 / 采样 / loader |

## 快速开始

```bash
cd vllm-v1
source ../vllm-v0/.venv/bin/activate   # 或共用仓库 venv

python run.py --model Qwen3-0.6B --prompt "Hello" --max-new-tokens 64 \
  --temperature 0 --prefix-backend hash

# 第二次同前缀应看到 cached_prompt_tokens > 0
python run.py --model Qwen3-0.6B --prompt "Hello" --max-new-tokens 64 \
  --temperature 0 --prefix-backend radix

python scripts/check_prefix_hit.py --model Qwen3-0.6B --backend hash
python scripts/check_prefix_hit.py --model Qwen3-0.6B --backend radix
```

`--prefix-backend`：`none`（不跨请求共享）| `hash` | `radix`。

## 设计文档

详见 [docs/kv_cache.md](docs/kv_cache.md)。v0 讲义仍在 [`../vllm-v0/docs/`](../vllm-v0/docs/)。
