# vllm-v1 相对 v0：新增与改写清单

源目录：`../vllm-v0/`。本文件只列 **差异**，便于对照教学。

## 新增（v0 没有）

| 路径 | 说明 |
| --- | --- |
| `src/kv/` | KV 独立包 |
| `src/kv/__init__.py` | 公开 API |
| `src/kv/context.py` | `ForwardContext`（页表契约） |
| `src/kv/sequence.py` | 轻量 `Sequence` |
| `src/kv/pool.py` | `_Block` / `BlockPool` |
| `src/kv/manager.py` | `KVManager` |
| `src/kv/prefix/noop.py` | `none` 后端 |
| `src/kv/prefix/hash_util.py` | `compute_block_hash`（blake2b） |
| `src/kv/prefix/hash_index.py` | hash 前缀索引 |
| `src/kv/prefix/radix_tree.py` | 压缩 radix 树 |
| `src/kv/prefix/radix_index.py` | radix 前缀索引 |
| `src/pagedattention/` | **PagedAttention**：slots + `PagedStore` |
| `src/pagedattention/slots.py` | 公开 `slots_tensor`；内部 `_slot_of` / `_slots_for_range` |
| `src/pagedattention/store.py` | scatter-write / gather-read |
| `test/test_pagedattention.py` | 页表读写单测 |
| `test/test_e2e_smoke.py` | Engine 冒烟 |
| `scripts/check_prefix_hit.py` | 前缀命中回归 |
| `docs/ADDED_vs_v0.md` | 本清单 |
| `docs/kv_cache.md` | KV / PagedAttention 设计说明 |

## 从 v0 同步（内容基本一致）

| 路径 |
| --- |
| `src/sampler.py` |
| `src/model_loader.py` |
| `src/model_hub.py` |
| `src/layers/activation.py` |
| `src/layers/layernorm.py` |
| `src/layers/rotary.py` |

## 基于 v0 改写

| 路径 | 变化 |
| --- | --- |
| `src/config.py` | +`prefix_backend` / `block_size` / `num_kv_blocks` |
| `src/model.py` | Attention 传入 `layer_id` |
| `src/engine.py` | prefill+decode；组装 `slot_mapping` + `KVManager` |
| `src/layers/attention.py` | `PagedStore.write/gather` + SDPA（不依赖 Manager） |
| `run.py` | +`--prefix-backend` / `--block-size`，打印 `cached_prompt_tokens` |
| `README.md` | 反映 v1 KV / 前缀能力 |
