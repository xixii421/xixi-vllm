# v1 KV 设计（block 池 + hash / radix）

## 相对 v0 多了什么

v0 每步重算整段序列，无 KV。v1：

1. **物理 KV**：按 page（block）存放 `[layers, blocks, block_size, kv_heads, dim]`
2. **前缀索引**：跨 `generate()` 复用满块前缀（可选）
3. **Attention**：write 进池 → gather 成连续 → SDPA（无 FlashInfer）

## 目录

```text
src/kv/
  context.py      ForwardContext（manager + seq + cache_len）
  sequence.py     token_ids / block_table / num_cached_tokens
  pool.py         Block + BlockPool
  gather.py       write_kv / gather_kv
  manager.py      allocate / ensure_blocks / finish
  prefix/
    noop.py       none：永不命中
    hash_index.py 链式 blake2b → block_id
    radix_tree.py 压缩基数树
    radix_index.py 树上的前缀匹配
```

## 数据流

```text
prompt → Sequence → KVManager.allocate
                      ├─ index.match → 命中满块（hash 或 radix）
                      └─ 其余 allocate_fresh
        → Prefill 仅 token_ids[num_cached:]
        → Decode 逐步 append + ensure_blocks
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

Continuous batching、Scheduler、CUDA Graph、FlashInfer PagedAttention、多卡。
