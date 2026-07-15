# vllm-v2 — PagedAttention

在 v1「RoPE 后写 Cache、Decode 只喂新 token」语义上，把连续 KV 换成 **Block 池 + block_table + slot_mapping**。Continuous Batching / Scheduler 留给 v3。

## 源码分层

```text
src/
  config.py          # EngineConfig
  layers/            # Attention / RMSNorm / RoPE / Act   （算子）
  models/            # Qwen3 + loader + hub               （模型）
  memory/            # BlockManager / PagedKVCache / slot （KV 分页）
  engine/            # Engine.generate + Sampler          （编排）
  utils/             # ForwardContext（桥接 Engine ↔ Attention）
```

依赖方向：**engine → models / memory / utils → layers**。上层不反向依赖下层细节以外的包。

| 包 | 入口 | 作用 |
| --- | --- | --- |
| `src.memory` | `BlockManager`, `PagedKVCache`, `slots_for_range` | PagedAttention 存储与页表 |
| `src.models` | `Qwen3ForCausalLM`, `load_model`, hub 工具 | 权重与网络 |
| `src.engine` | `Engine` | Prefill/Decode 生命周期 |
| `src.layers` | `Attention`, … | 层实现（读 `ForwardContext`） |
| `src` | `Engine`, `EngineConfig` | 对外最简 API |

```python
from src import Engine, EngineConfig
# 或分层导入：
from src.memory import BlockManager, PagedKVCache
from src.models import Qwen3ForCausalLM
```

## 版本定位

| | v1 | v2 |
| --- | --- | --- |
| 要解决的问题 | 单请求重复计算 | 显存碎片 / 按需占用 |
| KV 布局 | `[L, max_len, H, D]` 连续 | `[2, L, N_blocks, B, H, D]` 分页池 |
| 映射 | 无 | `block_table` + `slot_mapping` |
| Attention | 切片读 | scatter 写 + gather 读（教学） |
| 调度 | 无 | 仍无 |

## 快速开始

```bash
cd vllm-v2
source ../vllm-v0/.venv/bin/activate

python run.py --model Qwen3-0.6B --prompt "Hello" --max-new-tokens 64 --temperature 0

python test/test_block_manager.py
python test/test_paged_kv.py
python test/test_paged_vs_v1.py --model Qwen3-0.6B --max-new-tokens 32

python -m bench.run_all --suite kv_vram --model Qwen3-1.7B --context-len 4096
```

## 设计文档

- **本版主题**：[docs/paged_attention.md](docs/paged_attention.md)
- **继承自 v1**：[docs/kv_cache.md](docs/kv_cache.md)

## 参照

- `nano-vllm`：`engine/` + `models/` + `layers/`（分层参照）；BlockManager / block_table（机制主参照）
- `ullm`：token 级 slot + FlashInfer Paged（page_size=1）
- 上游 `vllm`：`BlockPool` / `BlockTable` / `slot_mapping`
