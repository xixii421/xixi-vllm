# vllm-v3：调度与 Continuous Batching

本版相对 v2 的核心变化不是再改 KV 的物理布局，而是**把单请求串行 `generate` 升级成可插队的调度系统**。KV 仍然是分页池；新增 Scheduler、ModelRunner、带状态机的 Sequence，以及块级哈希前缀缓存。全程纯 PyTorch，不算子优化。

设计参考 `ullm` 的调度文档（waiting / running、状态转移、step），前缀缓存采用 vLLM / nano-vllm 的**链式块哈希**，不用 Radix。

---

## 1. v3 相对 v2 解决什么问题

v2 已经能按页分配 KV，但 `Engine.generate` 一次只服务一个请求。多个请求只能排队，Decode 每步只算 1 个 token，GPU 经常闲着；分页省下的显存也没有变成并发吞吐。

v3 引入 Continuous Batching（持续批处理）：

- 新请求随时进入 `waiting`
- 一步里可以同时推进多个 Decode（真正组 batch）
- Prefill-first：有等待中的 Prefill 就先做 Prefill，否则做 Decode
- **同一步不混合 Prefill 与 Decode**（教学上更清晰）

一句话：**v2 把 KV 存对；v3 在存对的基础上把多请求调度进来，并让相同前缀的整页 KV 能被共享。**

---

## 2. 模块结构

```text
src/
  sampling_params.py     每请求采样参数
  memory/
    sequence.py          Sequence 状态机 + slot_of
    block_manager.py     发页 / 还页 / 链式哈希前缀 + ref_count
    kv_cache.py          共享物理池 write / gather
  engine/
    scheduler.py         waiting / running 队列，schedule / postprocess / preempt
    model_runner.py      Prefill/Decode 组张量、跑模型、采样
    engine.py            add_request + step() + generate / generate_batch
    sampler.py           采样
  utils/context.py       ForwardContext（批次元数据）
  layers/attention.py    write → Prefill SDPA 或 Decode gather+SDPA
```

推荐阅读顺序：`Sequence` → `Scheduler` → `BlockManager`（哈希）→ `ModelRunner` → `Engine.step` → `Attention`。

---

## 3. Sequence 生命周期

每个请求是一个 `Sequence`，状态只有三种：

| 状态 | 含义 |
| --- | --- |
| WAITING | 在 waiting 队列，尚未完成 Prefill（含被抢占后重排队） |
| RUNNING | Prefill 完成，在 running 队列里做 Decode |
| FINISHED | 触达 EOS 或 `max_tokens`，页已 `deallocate` |

关键游标：

- `num_cached_tokens`：池里已经有效的 KV 长度（前缀命中会直接抬高它）
- `num_scheduled_tokens`：本步要算多少 token（Prefill ≥1，Decode =1）
- `block_table`：逻辑页 → 物理页

状态转移（与 `ullm` 文档同构，简化版）：

```text
ADD → WAITING
WAITING → RUNNING     SCHEDULE(prefill 完成)
RUNNING → WAITING     PREEMPT（释放页，重算）
RUNNING → FINISHED    采样到 EOS / 达到长度
```

---

## 4. Scheduler：控制面

`Scheduler` 只做整数决策，不碰显存里的浮点。

### 4.1 两队列

- `waiting`：新请求、未做完 Prefill 的请求、被 preempt 回来的请求
- `running`：已进入 Decode 的请求

### 4.2 `schedule()` 策略（Prefill-first）

1. 若 `waiting` 非空：从队头取，直到触达 `max_num_seqs` 或 `max_num_batched_tokens`
2. Prefill 时调用 `BlockManager.can_allocate` / `allocate`（内部做哈希最长前缀匹配）
3. 若本步调度到了任何 Prefill，立刻返回 `(seqs, is_prefill=True)`
4. 否则从 `running` 取 Decode：每序列 `num_scheduled_tokens=1`，不够页则 `preempt`
5. Decode 返回 `(seqs, is_prefill=False)`

### 4.3 `postprocess()`

1. `hash_blocks`：把本步新写满的页登记进哈希表
2. `num_cached_tokens += num_scheduled_tokens`
3. Prefill 未完成（切块 Prefill）则不追加采样 token
4. 否则 `append_token`；若 EOS / 超长则 `FINISHED` 并 `deallocate`

### 4.4 Preempt

页池不够时，从 `running` 尾部踢出一个请求：`deallocate` 清页表，状态回到 WAITING，插回 `waiting` 队头。这是 **recompute**，不是换出到 CPU。

---

## 5. 哈希前缀（不是 Radix）

粒度是 **整页 `block_size`**。每个写满的页算链式指纹：

```text
h_i = hash(token_ids_of_block_i, prefix=h_{i-1})
```

全局表 `hash → physical_block_id`。新请求 Prefill 前从左往右匹配满页：命中则 `ref_count += 1` 复用该页，并把 `num_cached_tokens` 抬到 `命中页数 × B`；只对后缀做真正 Attention。

与 Radix（`ullm` / SGLang）的差别：

| | 本版哈希块缓存 | Radix |
| --- | --- | --- |
| 粒度 | 整页 | 任意 token，可分裂节点 |
| 结构 | `dict[hash→block]` | 树 + LRU 驱逐 |
| 与 v2 | BlockManager 自然扩展 | 另一套 Cache 叙事 |

未写满的最后一页不入哈希表，避免半页污染。

---

## 6. ModelRunner：数据面接口

`ModelRunner.run(seqs, is_prefill)`：

- **Prefill**：教学实现里对本步调度到的序列**逐个** forward（调度已是持续批；Prefill 组变长批留给以后）。支持前缀命中：只喂后缀 token，Attention 里 gather 全前缀再做平移因果 mask。
- **Decode**：真正 `[B, 1]` 组 batch，`slot_mapping` 长度 B；Attention 对每条 gather 前缀后 SDPA。

采样在 CPU 标量路径上调用 `Sampler`（与 v2 相同）。

---

## 7. Engine.step

```text
outputs, metric = engine.step()
# schedule → model_runner.run → scheduler.postprocess
```

- `generate(prompt)`：塞一个请求后循环 `step`，流式吐 token（兼容 v2 用法）
- `generate_batch(prompts)`：一次塞入多个请求，靠 Scheduler 交错 Prefill/Decode，返回按输入顺序的文本

---

## 8. CPU / GPU 分工（延续 v2）

| CPU | GPU |
| --- | --- |
| Scheduler 队列与页号 | `PagedKVCache` 池 |
| 哈希匹配 / ref_count | `write` / `gather` / SDPA |
| `slot_mapping` 列表算术 | 模型权重前向 |
| `SamplingParams`、tokenizer | |

「管理显存」仍然是管理页号；物理池在初始化时 `torch.zeros(device=cuda)` 一次分配。

---

## 9. 本版明确不做

- FlashInfer / 自定义 PagedAttention kernel / CUDA Graph → v4
- Radix 前缀 → 更后版本
- Prefill 与 Decode 混合进同一次 forward
- TP / PP / 异步调度 / OpenAI serving

---

## 10. 怎么跑

```bash
cd vllm-v3
source ../vllm-v0/.venv/bin/activate   # 或本目录 venv

python test/test_block_manager.py
python test/test_scheduler.py

python run.py --model Qwen3-0.6B --prompt "Hello" --max-new-tokens 64 --temperature 0
python run.py --model Qwen3-0.6B --batch --max-new-tokens 32
```

设计对照：`ullm/docs/scheduler_design.md`，前缀实现对照 `nano-vllm` 的 `BlockManager`。
