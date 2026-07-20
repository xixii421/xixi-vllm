# v1 KV 设计（PagedAttention 存储 + hash / radix 前缀）

架构总览与类图 / 时序 / 页状态机见 [architecture.md](./architecture.md)。

## 相对 v0 多了什么

v0 每步重算整段序列，无 KV。v1：

1. **物理 KV**：按 page（block）存放 `[layers, blocks, block_size, kv_heads, dim]`
2. **PagedAttention 契约**：`slot_mapping` 写、`block_table` 读（教学 gather→SDPA）
3. **前缀索引**：跨 `generate()` 复用满块前缀（可选）

## 高内聚分包

```text
src/pagedattention/          # 纯页表读写（不感知前缀 / Sequence）
  slots.py                   # 公开 slots_tensor；内部 _slot_of / _slots_for_range
  store.py                   # 公开 PagedStore.write / gather；内部 Protocol/helper

src/kv/                      # 分配生命周期 + 前缀索引 + ForwardContext
  context.py                 # store + slot_mapping + block_table
  sequence.py                # token_ids / block_table / num_cached_tokens
  pool.py                    # _Block + BlockPool（物理页 + refcount）
  manager.py                 # allocate / ensure_blocks / finish
  prefix/                    # none | hash | radix
```

依赖方向：

```text
Engine → kv(KVManager, ForwardContext) → pagedattention(PagedStore)
Attention → ForwardContext（只碰 store / slot_mapping / block_table）
```

Attention **不**依赖 `KVManager` / `Sequence`。读写只走 ``PagedStore``，无兼容包装层。

## 数据流

```text
prompt → Sequence → KVManager.allocate
                      ├─ index.match → 命中满块（hash 或 radix）
                      └─ 其余 allocate_fresh
        → Engine 生成 slot_mapping = slots[cache_len:write_end)
        → Attention: store.write(slots) → store.gather(block_table) → SDPA
        → Decode 逐步 append + ensure_blocks + 单 token slot
        → finish：publish 满块 + 页进入 cached（供下次命中）
```

## hash vs radix

| | hash | radix |
| --- | --- | --- |
| 结构 | `dict[block_hash → block_id]` | 压缩前缀树（key=块 hash 序列） |
| 匹配 | 从左线性扫满块直到 miss | `match_prefix` 一次走出共享前缀 |
| 适用 | 实现最短、易调试 | 多分支共享前缀时查找更干净 |
| 共同点 | 都以 **满 block** 为单位；尾块不共享 |

`prefix_backend=none` 时索引为 `NoopPrefixIndex`，行为退化为「仅本请求内的 paged KV」。

## Engine 要点

- **不要**每次 `generate` 把整池 `zero_()`，否则前缀无效
- Prefill 有前缀命中时，Q 只含未缓存段，K/V gather 全长，并用位置 mask 做因果
- `last_num_cached_tokens` 便于 CLI / 测试观察命中

## 非目标（留给后续 fork）

Continuous batching、Scheduler、CUDA Graph、FlashInfer 真·PagedAttention kernel（内核内按页表打分、无全量 gather）、多卡。
