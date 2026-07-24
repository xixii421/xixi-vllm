# 前缀缓存（Hash / Radix）

v3 资源层对齐 v2，并补齐 v2 radix 缺口；哈希热路径用 **xxhash**，页池用 **LRU cached**（参考 vLLM v1 APC）。

## 布局

```text
src/memory/prefix/
  hash_util.py     链式块哈希（xxhash）
  hash_index.py    dict[hash → block_id]
  radix_tree.py    压缩前缀树 + leaf LRU evict + detach invalidate
  radix_index.py   match / publish（含 dedup）/ pin / release
  noop.py          关闭前缀
src/memory/block_manager.py   free / used / cached（OrderedDict LRU）
src/memory/manager.py         KVManager：_fit_prefix / allocate / sync / deallocate
```

## 相对 v2

| 能力 | v2 | v3 |
| --- | --- | --- |
| `none` / `hash` / `radix` | 有 | 有（`EngineConfig.prefix_backend`） |
| Hash 生命周期 | cached 池 + invalidate | 同左 + LRU 驱逐 |
| Radix `invalidate` | no-op | **detach 零引用节点** |
| Radix eviction | 无 | **leaf LRU `tree.evict`** |
| Radix publish dedup | 无 | **共享前缀 remap 去重** |
| 哈希算法 | blake2b | **xxhash64**（更快） |
| Chunked sync | advance → publish full | **同 v2 合同** |
| 未完成 chunk sample | 仍采样 | **可跳过 LM-head** |

## 验证

```bash
cd vllm-v3 && PYTHONPATH=. pytest tests/test_prefix.py tests/test_scheduler.py -q
cd .. && python -m bench.prefix_compare --model Qwen3-0.6B --backends hash,radix
# or lean step-by-step:
python bench/_prefix_verify.py
```

### Full-hit promote

整页命中（`num_cached == num_tokens`）时必须 **promote → decode**，不能留在 waiting。
v3 已实现；v2 原先 `need <= 0` 直接 break 会卡死，已对齐补上 `_promote_full_hit`。

### Hit 指标

`Engine.last_num_cached_tokens` 取自 **`allocate()` 时的前缀命中**（`kv.last_num_cached_tokens`），
不再用 prefill 推进后的 `num_cached`（否则 miss 也会显示成 prompt 长度）。
