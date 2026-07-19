# vllm-v2 架构：三面分离 + Continuous Batching

## 相对 v1 多了什么

v1：单请求 `generate`、分页 KV、前缀缓存、gather→SDPA。  
v2：在同一套 KV/前缀/PagedAttention 上挂上 **Continuous Batching**。

```text
add_request → waiting
Engine.step():
  Scheduler.schedule()     # 控制面：Prefill-first 或 Decode 合批
  ModelRunner.run()        # 数据面：组张量 + ForwardContext
  Scheduler.postprocess()  # 控制面：推进游标 / EOS / 还页
```

## 三面依赖

```text
控制面 Scheduler ──决策整数──► KVManager（分配/扩页/还页）
数据面 ModelRunner ──张量──► Qwen3 + Attention
计算面 Attention  ──只读──► ForwardContext(store, slots, block_tables)
```

Attention **不**依赖 Scheduler / Sequence。

## 教学边界（与旧 v3 一致）

- Prefill 调度可多请求，前向仍串行（`run_prefill_one`）
- Decode 组 `[B,1]`；Attention 按序列 gather+SDPA 循环
- Prefill 与 Decode **不同一步混合**
- Preempt = recompute（不 publish 前缀）
- 前缀继续复用 v1 的 `none|hash|radix`
