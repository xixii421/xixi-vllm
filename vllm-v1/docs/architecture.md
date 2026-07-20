# vllm-v1
> **核心设计**：KV Management

> [上一节](../README.md) | [下一节](./ADDED_vs_v0.md)

v0 解决的是怎样把 Qwen3 从权重到逐字输出这条链路跑通。

v1 的核心是把历史 Token 的 Key / Value 存下来，并让这条缓存真正接到 Engine 与 Model 上。


## 1. 回顾 v0
v0 的自回归循环可以压成五步：

```text
拼整段序列 → 整段 forward → 取最后一位 logits → 采样 → append → 再来一轮
```

每一步模型看到的都是**从第一个 prompt token 到当前最后一个 token**的完整输入。Attention 里 Q/K/V 都是当场算出来的，算完即丢。结果是：
| 现象 | 后果 |
| :--- | :--- |
| 第 N 步仍要算前 N−1 个历史位置的 K/V | 计算量随长度平方涨 |
| 没有跨请求状态 | 两次 `generate` 即使 prompt 相同，也各算各的 |


## 2. KV Cache

在学习vllm-v1版本的时候，最核心的是思考这4个问题：
1. 没有 Cache 时缺什么
2. Cache 到底优化什么
3. 我们按什么理念设计
4. 最终换来什么效果。

### 2.1 当前 V0 缺点

自回归里，第 $t$ 步要预测下一个 token，Attention 必须看见位置 $0\ldots t-1$ 的 Key / Value。这件事数学上没问题。v0 也确实是这么实现的：
**每一步都把整段序列重新送进模型，让每一层现场重算全部历史的 K/V。**

但是，这会直接带来两类问题：
1. **计算重复**。生成第 $N$ 个新 token 时，前面 $N-1$ 个位置的 K/V 在上一步其实已经算过了，但因为没有地方存放，下一步只能再算一遍。decode 的 Attention 工作量随长度近似按 $O(N^2)$ 涨，长上下文时延迟暴涨。
2. **显存碎片化**。如果按最大输出长度给每条请求预留一整段连续 KV，这倒是可以消除重算，但长度不可预知时会大量预留、碎片化。KV缓存不能只做一个大 tensor 粘在模型上，而是应该设计一个类似内存池的东西，高效的对整个池子进行管理。

所以 v1 要同时解决两件事：**缓存KV**，以及 **管理KV**。

### 2.2 KV Cache 的核心作用
KV Cache可以让模型在输出过程中，每个 token 的 K/V 只算一次；之后每一步生成只负责算新 token 的 K/V 并追加，计算Attention的时候 再把缓存里的历史 + 当前生成的新token拼起来做注意力。

对应到两阶段：

- **Prefill**：把 prompt尚未在缓存里的那一段过一遍模型，把得到的 K/V 写入缓存。这一步仍可能处理较长序列，但是一次性计算。
- **Decode**：每一轮模型输入长度变为 **1**。新算出的 K/V 添加进缓存；做 Attention 时 读出长度为当前总序列的 K/V，再和长度为 1 的 Q 做 SDPA。

因此 Cache 的第一性收益是 **算力**。decode 从每步重算整段，变成每步只算增量。  
第二性收益是 **显存形态**与 **跨请求复用**。页式存放可以减少预留浪费；前缀索引让相同 prompt 前缀不必再 prefill 一遍。

### 2.3 KV 管理的设计理念

明确了要缓存 K/V，还要管得住显存之后，真正难的其实不是实现，而是怎么把这一套设计划分好任务边界。如果把所有事塞进 `Attention`，或塞进一个超级 `Manager`，后面任何改动都会牵一发而动全身。
因此我在这里的设计方案是先想清楚**三条理念**：

我用一个例子来带大家体会一下KV Management的逻辑设计，北邮旁边有一个金谷园饺子店，有很多桌位，人们会来这里吃饺子。


**理念一：逻辑与物理分开**
从顾客的角度来说，每一个人拿到的都是一个连续的号码，例如A01、A02等号码，但是实际上坐到的地方可能是33号、68号桌子等等，物理位置和逻辑位置是分离的。
顾客只需要知道自己是逻辑第几位就可以了，并不关心桌子在哪，对照着表来映射即可。
同理，操作系统管理内存的关键，并不是把一块巨大的 RAM 直接裸给进程，而是让进程看见连续的虚拟地址，每一个进程都有完整的地址映射空间，但是硬件上是一页一页不连续的物理页，中间靠页表翻译。
KV 也一样。一条请求在逻辑上拥有从第 0 个 token 往后连续增长的地址空间；GPU 上则是预先划好的**固定大小页池**，请求变长就多拿几页，用完再还。请求不需要知道某页落在显存的哪一片连续区间。

**理念二：管资源与做存储分开。**  
在金谷园有一个非常特殊的岗位，门口查看监控的阿姨。她实际上是在管理资源，谁来了坐哪张桌子？是坐小桌子还是大桌子？前边一桌的人是不是已经吃完空出来了等等？这个其实是资源管理。

而还有保洁员，需要对上一桌子的残羹剩饭进行清理，以及摆放新的餐具，并且通知给前台阿姨。这个阿姨是实际上在管理各个桌子的”读写“。
那么从KV的角度来说，一旦有了页，就会分成两类问题。第一类偏**资源与生命周期**：这页现在归谁，前缀能不能复用，请求结束是留下还是回收？第二类偏**存储怎么访问**：已经知道页号之后，怎样把算出来的 K/V 写进张量，怎样按逻辑顺序读回来给 Attention？对应到代码，KVManager就是前台，PagedStore就是保洁员。

**理念三：编排与计算分开。** 
保洁不需要知道来了多少人，前台也并不关心保洁的工作，顾客更是不担心桌子是否收拾好，因此它们都是有权责边界的。
并且它们之间是有状态切换和先后顺序调度的，因此Engine就需要管理，什么时候告诉保洁、什么时候告诉顾客，什么时候更新前台信息等等。
因此需要第三层关系：**编排层告诉计算层这一步的规则，计算层按规则读写 Cache。** 



### 2.4 效果
1. **Decode 计算量下降**。模型每步输入从长度 $N$ 变为长度 $1$；历史 K/V 来自 Cache。这是提升 tok/s 的主因。
2. **显存按需占页**。短请求少占页；结束后满块可留作前缀复用，其余回池。同一块物理显存可被多轮请求轮转，语义上也对齐后续 Continuous Batching。
3. **模型骨架几乎不变**。变的是 Engine 喂什么、Attention 是否按规则走 Cache。排错边界清晰：资源问题找 `kv`，读写问题找 `pagedattention`，算子问题找 `Attention`。
4. **前缀命中是叠加项**。打开 hash/radix 可跳过已缓存满块的 Prefill；关掉则退化为单请求内的分页 KV，主路径不变。

## 三、计算流程
下面按一次 `Engine.generate` 走完整路径。

### 3.1 启动时多出来了什么
相比较v0，Engine会构造一个KVManager：
```text
KVManager
  ├─ BlockPool     物理 k/v + free/used/cached 三态
  ├─ PagedStore    绑在 pool 上的写/读 API
  └─ PrefixIndex   none | hash | radix（可选）
```

模型权重没有任何变化，只是在涉及到kV的时候，从计算变成了访存读取。

### 3.2 请求进入

```text
prompt
  → tokenizer.encode → input_ids
  → kv.new_sequence(input_ids)   得到 Sequence（token_ids，尚无可靠页表）
  → kv.allocate(seq)
        ├─ 问 PrefixIndex：满块前缀能否命中已有物理页？
        ├─ 命中：reclaim / 加引用，写入 block_table 前缀
        └─ 未命中部分：allocate_fresh 新页接到 block_table 后面
  → seq.num_cached_tokens = 命中满块 × block_size
```

此时还没有进行计算，先进行映射：**这条请求已经知道，逻辑上第几页对应物理上哪一块。**

### 3.3 Prefill
**A. 未命中或只部分命中**

```text
Engine._bind_ctx(seq, is_prefill=True, cache_len=cached, write_end=prompt_len)
  → 用 block_table 生成 slot_mapping（只覆盖要新写入的那段）
  → set_forward_context(ForwardContext(...))

模型输入只给未缓存段：
  ids       = input_ids[cached : prompt_len]
  positions = cached … prompt_len-1

model.forward → 各层 Attention：
  算出本段 K/V → store.write(slot_mapping)
  store.gather(整段 0…prompt_len) → 与当前 Q 做 SDPA
  （若有前缀：Q 短、K/V 长，用位置 mask 保因果）

取末位 hidden → logits → 采出第一个生成 token
```

**B. 整段 prompt 都在前缀缓存里**

历史上的 K/V 都在，但最后一位的 hidden / logits 仍要落到下一个生成 token。v1 会退成：只对最后一个 prompt token 再走一步类似 decode 的 forward，然后采样。

无论 A/B，**Prefill 结束后，pool 里已经放好了这条请求当前所需的物理 KV。**

### 3.4 Decode

```text
seq.append_token(next_id)
kv.ensure_blocks(seq)          # 可能跨过页边界，需要新物理页

循环：
  _bind_ctx(cache_len → cache_len+1)   # slot_mapping 只有 1 个 slot
  ids = [[next_id]]
  positions = [[cache_len]]
  model.forward
      Attention: write 这一步的 K/V → gather 全长 → SDPA（通常 q_len=1）
  logits → sample → yield
  append + ensure_blocks
  cache_len += 1
```

### 3.5 收尾

```text
finally:
  set_forward_context(None)
  kv.finish(seq)
      ├─ 把满块的元数据 publish 进 PrefixIndex（若开启）
      ├─ 满块页 → cached（供下次 match）
      └─ 尾块未满 / 或 none 后端 → free
```

下一次 `generate`，若 prompt 共享相同满块前缀，`allocate` 就能直接复用那些 `cached` 页，Prefill 跳过对应 token——这是 v1 的第二层收益（跨请求），建立在第一层（页内 KV 缓存）之上。

---

## 四、KV 如何并入 Engine：编排职责对照

可以把 Engine 理解成导演，KV 是场务，Model 是演员。

```text
run.py
  └─ Engine
       ├─ tokenizer / model / sampler     （与 v0 相同角色）
       ├─ kv: KVManager                   （v1 新增：场务）
       └─ generate()
            ├─ kv.new_sequence / allocate
            ├─ 每步 _bind_ctx + model(...) + sample
            ├─ kv.ensure_blocks
            └─ finally: kv.finish
```

Engine 相对 v0 **改写的部分**，说到底只有这些：

1. **循环拆成 Prefill + Decode**，不再每步喂整段 `input_ids`。
2. **每步 forward 前绑定 ForwardContext**（页表合同）。
3. **用 Sequence 管理 token 与 block_table**，而不是只拿一个 Python list。
4. **结束时 finish**，把页退回池子 / 交给前缀索引。

Engine **仍然不实现** Attention 数学，也 **不直接** `index_copy_` 物理张量；它只调用 `KVManager` 与 `slots_tensor`，再调 `model`。

---

## 五、KV 如何并入 Model：只改 Attention 契约

模型堆叠方式与 v0 一致：`Embedding → N × DecoderLayer → Norm → lm_head`。  
Decoder 里仍是 RMSNorm → QKV → RoPE → **Attention** → 残差 → MLP。

v1 真正改掉的模型侧文件，几乎就是 `layers/attention.py`：

```text
v0 Attention.forward(q, k, v):
    当场 SDPA(q, k, v)          # 历史全在这一次的 k/v 里

v1 Attention.forward(q, k, v):
    ctx = get_forward_context()
    if ctx is None:
        退回 v0 行为（无缓存路径）
    else:
        store.write(本层, slot_mapping, k, v)   # 追加进物理池
        k_c, v_c = store.gather(本层, block_table, seq_len)
        SDPA(q, k_c, v_c)                       # 历史来自缓存
```

因此：

- **上层 `Qwen3ForCausalLM.forward(ids, positions)` 签名可以不变**，仍然表示当前这批 token 的 id 与位置。
- **变的是 ids/positions 的语义**：v0 是整段；v1 Prefill 是未缓存段，Decode 是单 token。
- **Attention 从自给自足，变成读合同**；合同由 Engine 写好，与层编号 `layer_id` 一起决定写到池的哪一层。

这就是融合方式的全部要点：**Model 不算生命周期，只认本步合同；合同由 Engine 在调用 Model 之前放进线程局部上下文。**

---

## 六、一次完整调用的路径（总图）

```text
EngineConfig + 权重
    │
    ▼ Engine.__init__
    │   tokenizer / model / sampler
    │   KVManager → BlockPool + PagedStore + PrefixIndex
    │
    ▼ engine.generate(prompt)
    │
    │  ┌─ new_sequence + allocate ─┐
    │  │  Sequence.block_table     │
    │  │  num_cached_tokens        │
    │  └───────────────────────────┘
    │
    │  ┌─ Prefill ──────────────────────────────┐
    │  │  _bind_ctx → set_forward_context       │
    │  │  model(未缓存 ids, 对应 positions)      │
    │  │     Attention: write → gather → SDPA   │
    │  │  sample → 第一个新 token               │
    │  └───────────────────────────────────────┘
    │
    │  ┌─ Decode × N ───────────────────────────┐
    │  │  append + ensure_blocks                │
    │  │  _bind_ctx（单 slot）                   │
    │  │  model([[token]], [[pos]])             │
    │  │     Attention: write → gather → SDPA   │
    │  │  sample → yield                        │
    │  └───────────────────────────────────────┘
    │
    ▼ finally: clear context + finish(seq)
         满块 → PrefixIndex + cached；尾块 → free
```

读代码时建议按这个顺序跟：

1. `src/engine.py` 的 `generate` / `_bind_ctx`（编排）
2. `src/kv/manager.py` 的 `allocate` / `ensure_blocks` / `finish`（页生命周期）
3. `src/layers/attention.py` 的 `forward`（合同消费）
4. `src/pagedattention/store.py` 的 `write` / `gather`（物理读写）
5. 需要时再进 `src/kv/prefix/*`（跨请求）

---

## 七、前缀缓存在整条故事里站在哪

前缀不是另一套管线，而是挂在 **allocate / finish** 两端的加速器：

```text
请求结束 finish ——publish→ 索引记住：这些满块 token 对应哪些物理页
下一次 allocate ——match→  直接拿回物理页，Prefill 跳过对应长度
```

- 粒度是 **满 block**，尾块不共享（没写满就无法安全当公共前缀）。
- `none`：只有单次请求内的 KV Cache，无跨请求复用。
- `hash` / `radix`：索引结构不同，对 Engine / Attention 完全透明。

理解主流程时，可以先当成 allocate 总是全 miss。主链路通了再打开前缀后端。

---

## 八、v1 刻意还没做的事

| 能力 | v1 现状 | 留给后续 |
| :--- | :--- | :--- |
| Continuous Batching | 仍是单请求 `generate` | 调度器 + 迭代组 batch |
| 真·PagedAttention kernel | gather 成连续张量再 SDPA | FlashInfer / 内核内按页打分 |
| 多卡 / 异步服务 | 单卡同步 | TP、多进程 |
| 调度驱逐策略 | free 不够才挤 cached | 更完整的 eviction |

<span style="color:#b45309">**机制**</span> 先把主脊梁立住：历史 K/V 存在哪，谁负责分配，Model 怎么看见。吞吐向优化应落在调度与算子，而不是再把 Attention 与页管理耦死。

---

## 九、文件结构（相对 v0）

```text
vllm-v1/
├── run.py
├── docs/
│   ├── architecture.md      ← 本页（流程总览）
│   ├── kv_cache.md          分包与前缀对照（偏短）
│   └── ADDED_vs_v0.md       相对 v0 增改清单
└── src/
    ├── engine.py            编排：Prefill/Decode + bind ForwardContext
    ├── kv/                  页生命周期 + 前缀 + Context
    │   ├── manager.py
    │   ├── pool.py
    │   ├── sequence.py
    │   ├── context.py
    │   └── prefix/
    ├── pagedattention/      纯页表读写（写 slot / 读 block_table）
    ├── layers/attention.py  消费 ForwardContext
    └── …                    其余与 v0 对齐（model / sampler / …）
```

---

## 十、对照图（备查）

下面几张图与上文同一套逻辑，便于扫一眼回忆边界；细节仍以源码为准。

### 10.1 分层依赖

```mermaid
flowchart TB
  Eng["Engine"]
  M["Qwen3ForCausalLM"]
  Attn["Attention"]
  KVM["KVManager"]
  Ctx["ForwardContext"]
  Store["PagedStore"]
  Pool["BlockPool"]

  Eng --> KVM
  Eng --> Ctx
  Eng --> M
  M --> Attn
  Attn -->|get| Ctx
  Ctx -->|store / slots / table| Store
  KVM --> Pool
  KVM --> Store
  Store --> Pool
```

### 10.2 页的三态

```mermaid
stateDiagram-v2
  [*] --> free
  free --> used: allocate_fresh
  cached --> used: 前缀命中 reclaim
  used --> cached: finish 且满块 publish
  used --> free: finish 不保留
  cached --> free: 池不够时淘汰
```

### 10.3 单步合同如何进到 Attention

```mermaid
sequenceDiagram
  participant E as Engine
  participant C as ForwardContext
  participant M as Model
  participant A as Attention
  participant S as PagedStore

  E->>C: set(slot_mapping, block_table, store, …)
  E->>M: forward(ids, positions)
  M->>A: forward(q,k,v)
  A->>C: get()
  A->>S: write
  A->>S: gather
  A-->>M: attn out
  M-->>E: hidden → logits
```

---

## 十一、一句话收束

- **v0**：Model 每步看见整段历史，K/V 用完即弃。  
- **v1**：Engine 用 `KVManager` 管页、用 `ForwardContext` 把页表合同塞进当步；Model 的 Attention 按合同把 K/V 写入物理池并读回历史，从而 Prefill 写缓存、Decode 只算新 token。  
- **前缀**：只是把 finish 留下的满块页，在下一次 allocate 时再认领回来。

读完本文，应能不看代码说出一次 `generate` 里，谁在什么时候写 KV，谁在什么时候读 KV。再下钻时，用文末图加 `engine.py` → `attention.py` → `manager.py` → `store.py` 这条路径即可。
