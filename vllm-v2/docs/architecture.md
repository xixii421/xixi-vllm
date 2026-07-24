# vllm-v2 架构：三面分离 + Continuous Batching + Chunked Prefill

## 相对 v1 多了什么

v1：单请求 `generate`、分页 KV、前缀缓存、gather→SDPA。  
v2：在同一套 KV/前缀/PagedAttention 上挂上 **Continuous Batching**，并支持
**chunked prefill + Prefill/Decode 同一步混跑**（可关）。

```text
add_request → waiting
Engine.step():
  Scheduler.schedule()  → (prefills, decodes)
  ModelRunner.run()     # decode batch，再串行 prefill chunk
  Scheduler.postprocess # 分别推进两类序列
```

## 调度策略（`mix_prefill_decode`）

| 模式 | 行为 |
| --- | --- |
| `False`（legacy） | Prefill-first：一步要么全是 prefill，要么全是 decode |
| `True`（默认） | 有 running 时：**先 decode**，剩余 token 预算给 **1 个** prefill chunk；无 running 时仍打包 prefill |

`max_num_batched_tokens` 限制每步 token 数 → 长 prompt 自动切块。

## 三面依赖

```text
控制面 Scheduler ──决策整数──► KVManager（分配/扩页/还页）
数据面 ModelRunner ──张量──► Qwen3 + Attention
计算面 Attention  ──只读──► ForwardContext(store, slots, block_tables)
```

## 教学边界

- Prefill 前向仍串行（`run_prefill_one`）；混跑 = 同一步两次 forward，非 fused kernel
- Decode 组 `[B,1]`；Attention 按序列 gather+SDPA 循环
- Preempt = recompute（不 publish 前缀）
- 前缀继续复用 v1 的 `none|hash|radix`

## 验证

```bash
python test/test_scheduler.py
python -m bench.compare_chunked --model Qwen3-0.6B --budget 64 --conc 2,4
```
