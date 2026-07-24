# 04 · 前缀 APC（瓶颈 F：有效性能）

相对 v2，前缀不是「算子 FLOPs」主战场，但在 **KV 打满、多分支共享前缀** 时决定你是否还要重算 prefill。  
算子主线见 [01](./01-v2性能瓶颈.md) / [02](./02-v3如何消除瓶颈.md)。

---

## 1. v2 留下的问题（有效性能）

| 问题 | 后果 |
| --- | --- |
| Radix `invalidate` no-op | 页回收后索引脏 |
| 无树级驱逐 | 压力下难保住热前缀 |
| Full-hit `need<=0` 不 promote | waiting 卡死（像「极慢」） |
| Hash 用 blake2b | CPU 略慢（相对 GPU 可忽略） |

---

## 2. v3 补齐

```text
src/memory/prefix/
  hash_util.py / hash_index.py     xxhash + 索引
  radix_tree.py / radix_index.py   树 + leaf LRU + invalidate/dedup/pin
  noop.py
block_manager.py                   free / used / cached（hash：页 LRU）
manager.py                         allocate / sync_prefix / deallocate
```

| 能力 | hash | radix |
| --- | --- | --- |
| 命中 | 链式块哈希 | 压缩前缀树 |
| 压力驱逐 | **页池 OrderedDict LRU** | **树叶 `tree.evict` → allocate** |
| invalidate | 索引删 | detach 零引用节点 |
| publish dedup | — | 共享前缀 remap |

Chunked：与 v2 同合同（advance → `sync_prefix`）；未完成 chunk 可跳过 LM-head。  
Full-hit：promote → decode。

---

## 3. 怎么讲给学生

- **同命中**：hash ≈ radix 的 GPU 时间。  
- **池满挤页**：hash 可能打穿热前缀首页；radix 倾向丢冷叶、保热干。  
- 这是 **命中率策略**，不是 FlashInfer 快慢。

---

## 4. 验证

```bash
cd vllm-v3 && PYTHONPATH=. pytest tests/test_prefix.py tests/test_scheduler.py -q
cd .. && python bench/_prefix_verify.py
cd vllm-v3 && PYTHONPATH=. python -m bench.prefix_evict_compare
```

`Engine.last_num_cached_tokens` = **allocate 时**的命中，不是 prefill 推完后的 `num_cached`。
