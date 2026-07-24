# vllm-v1 的 KV Cache 详解
> 参考资料：[vllm调度器的细节](https://zhuanlan.zhihu.com/p/692276562) | 
## 总览

自回归生成时，每一步预测下一个 token，Attention 都必须看见到目前为止全部历史的 Key 和 Value。

v0 的做法很直接，每一步把整段序列重新送进模型，历史 K/V 算完就扔。这样一来，越往后算得越多，长上下文会特别慢。

v1 换了一种做法，每个 token 的 K/V 只算一次，先存进 GPU；下一步只算新 token，再把已经存好的历史按正确顺序取回来做 Attention。存本身不难，难的是显存放哪。输出长度事先不知道，多条请求还要共用一块显存，不能给每条请求预留一整段连续大缓冲。业界的办法是把 KV 显存切成固定大小的页，每条请求自己维护一张页表，记下逻辑上第几页对应物理池里哪一块。这就是常说的把虚拟内存用到 KV 上。

在代码中，我单独写了kv模块和pagedattention模块，这两个是相比较v1新增的，[新增模块示意](./02-相比v0的增改.md)

主要由以下这么几个模块：KV manager, Pool, Slots, Store, Sequence, Prefix等等，下边我来逐一讲解。
![KV Module Overview](./posts/01-kv-overview.png)

读模块之前，先建立一个总图。v1 把「缓存历史 K/V」拆成三层：

```text
┌─────────────────────────────────────────────────────────────┐
│  Engine（编排）                                               │
│  · 何时 allocate / ensure / finish                            │
│  · 每步喂多长 ids、positions                                  │
│  · 每步绑定 ForwardContext                                    │
└───────────────┬─────────────────────────────┬───────────────┘
                │                             │
                ▼                             ▼
┌───────────────────────────┐   ┌─────────────────────────────┐
│  KVManager（资源与生命周期） │   │  Model → Attention（计算）   │
│  · Sequence / block_table │   │  · 算本步 Q/K/V              │
│  · BlockPool 领页还页      │   │  · 按合同 write → gather     │
│  · PrefixIndex 跨请求复用  │   │  · 再 SDPA                   │
└─────────────┬─────────────┘   └──────────────▲──────────────┘
              │                                │
              │         PagedStore             │
              └────────（物理读写）─────────────┘
                        BlockPool.k / .v
```

Engine 决定这一步演什么；Manager 与页池决定页归谁；Store 按地址读写；Attention 只认本步契约。后文按**状态 → 物理池 → 寻址 → 读写 → 契约 → 计算 → 编排 → 引擎时间线**展开。

## Sequence：一条请求的逻辑状态
> 源码：[src/kv/sequence.py](../src/kv/sequence.py)

vllm本身对Sequence进行了充分的封装，它是一个独立的文本生成流，即一个从输入prompt到输出completion的完整过程，核心就是对数据与状态的封装。


![Sequence细节](./posts/05-Sequence细节.png)

这里它封装了Group等多种功能，但是我们在这里进行简化：
```py
@dataclass
class Sequence:
    """ 一条生成请求的 tokens 与物理页表。 """

    token_ids: list[int]                              # 已有 prompt + 已生成 token
    block_size: int                                   # 每页 token 数
    seq_id: int = 0                                   # 进程内唯一 id（radix 引用用）
    block_table: list[int] = field(default_factory=list)  # 逻辑页 → 物理块号
    num_cached_tokens: int = 0                        # 前缀命中跳过的 token 数

```

分页 KV 有两套地址。逻辑上 token 按 0、1、2… 连续编号；物理上对应的 K/V 却可能落在互不连续的页里。必须有人同时记住「这条请求有哪些 token」和「逻辑页对应哪块物理页」，v1 把这份状态收成一个对象，就是 Sequence。

Sequence 只记请求侧的逻辑信息，不存 K/V、不占显存、不算 Attention。引擎与管理器共用同一份活状态：创建时只有 prompt，页表为空；管理器填页表，前缀能复用则记下可跳过的长度；之后每步从 Sequence 抽出本步页表与长度，交给 Attention，采出的 token 再写回，跨页时请管理器补页；结束时交回管理器，页表清空。前缀索引可只读这份状态；Attention 与物理读写模块只看见本步合同，从不直接碰 Sequence。

> **为什么要单独成一个模块，而不是在引擎里散落几个变量？**
> 
> 因为 token 列表、页表、前缀跳过长度必须跟着同一条请求一起创建、变长、销毁，否则 Decode 一长就容易出现长度和页表对不上。

Sequence 是请求的逻辑户口本，引擎和管理器主用，前缀索引只读，Attention 不用；它把有哪些 token 和页表绑在同一生命周期上。


## Pool：物理页池
> 源码：[src/kv/pool.py](../src/kv/pool.py)

Sequence 解决了请求侧记什么，包括有哪些 token、页表怎么指。
但它手里没有真正的 K/V 物理地址。数据必须放在 GPU 显存里，而且不能给每条请求各开一段连续大缓冲。

为什么呢？因为请求长度事先不知道，请求走了一拨又一拨，显存得能拆开再拼、用完再还。
于是我们不能给每个请求一个大缓冲，而是需要一个全局共享的物理页池。
- 启动时按模型形状挖好固定数量的页
- 运行中只做领页、还页、必要时淘汰，不再反复申请释放大块显存。

本仓库里这个对象叫 `BlockPool`。

页池负责两样东西：一块（其实是两块）巨大的 k、v 张量；以及对于Pool中张量的管理。

代码中，k 和 v 形状都是五维：

```py
[num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

1. 第一维是层，因为每一层 Transformer 都有自己的一套 K/V；
2. 第二维是物理页号，从 0 数到池子大小减一；
3. 第三维是一页里能装几个 token；
4. 后两维是每个 token 上 KV head的布局。引擎启动时根据模型层数和头维一次性创建，页数由配置给定。

页主要有三种状态：free、used、cached。空闲可领的进 free，正在被某条请求占用的进 used，请求已经结束但内容先留着给以后前缀命中的进 cached。

领页时（`allocate_fresh`）：优先从 free 取一页放进 used，引用计数置 1；free 空了才从 cached 里淘汰一页（通知前缀索引作废），清掉元数据后当 free，再领成 used。前缀命中时（`reclaim_cached`）：把已有内容的 cached 页直接拉回 used，K/V 不用重算。

请求结束还页时（`finish` → `release_to_cached_or_free`），先减引用；若引用仍大于 0（还有别的序列共用），页继续停在 used。减到 0 之后才二选一。

也就是说：**cached 和 free 都是“当前没有请求在用”，差别在内容还值不值得留给下一趟。**  
cached：物理 K/V 还在，哈希/token 元数据还在，前缀索引还能指过来。  
free：元数据清空，只剩可再分配的槽位，里面的旧数随时会被新请求盖掉。

举个例子。设 `block_size=4`，某次请求最终有 10 个 token，页表是 `[3, 7, 1]`，并开了 hash 前缀：

```text
逻辑:  0 1 2 3 | 4 5 6 7 | 8 9
物理:     3    |    7    |  1
         满块     满块     尾块（未满）
```

`finish` 时 `num_full_blocks=2`，只会 `publish` 物理页 3 和 7。于是：

- 物理页 3、7 → **cached**（下次相同前缀可 match 复用）  
- 物理页 1 → **free**（尾块不能进前缀，内容作废）

若 `prefix_backend=none`，三页在引用减到 0 后**全部进 free**，不会留 cached。若 free 已经耗尽、又来新请求要领页，就会从 cached 里淘汰一页（例如挤掉 3），索引里对应映射删掉，这一页再变成 free 后被新请求写成 used。

而当前缀命中时，还有可能把一页从 cached 中重新变成 used，整个状态机如下图所示：
![Pool状态机](./posts/02-BlockPool状态机.png)


和页号一一对应的还有一份很轻的元数据（源码里的 _Block）：引用计数、块哈希、这一页对应过哪些 token。

而真正的数值始终在 pool.k 和 pool.v 里；元数据是给前缀索引和淘汰策略看的小纸条。启动时管理器会创建页池，并往往挂上淘汰回调，这样 cached 页被挤掉时索引不会继续指向已失效的物理块。



## Slots 逻辑位置如何映射为物理行号
> 源码见 [src/pagedattention/slots.py](../src/pagedattention/slots.py)

上边已经说明，页池提供了存放 K 与 V 的物理空间，Sequence 中的页表则记录了逻辑页号到物理块号的对应关系。仅有这两项信息仍不足以完成写入。模型给出的是按逻辑顺序编号的 token，而页池按物理行存放数据。因此还必须把每一个逻辑位置翻译成页池中的一个全局行号。本仓库将这个行号称为 slot。

该翻译与经典的虚实地址转换属于同一类问题。给定页大小与页表，对任意逻辑位置 pos，先用整除得到它所在的逻辑页号，再用取余得到页内偏移，最后用页表查出物理块号，并与页内偏移合成为 slot。实现上只依赖整数除法与取模，不涉及张量运算。
设页大小为 block_size，页表为 block_table。则有

```text
physical = block_table[pos // block_size]
offset   = pos % block_size
slot     = physical * block_size + offset
```

其中 physical 为物理块号，offset 为块内下标，slot 为展平后的全局行号。写入时以 slot 作为目标行索引。

下面用一个最小实例说明虚实分离。取页大小等于 2，页表等于 [1, 0]。于是逻辑页 0 映射到物理块 1，逻辑页 1 映射到物理块 0。四个逻辑位置的计算结果如下。

```text
逻辑位置      0    1  |  2    3
所在逻辑页    0    0  |  1    1
物理块号      1    1  |  0    0
页内偏移      0    1  |  0    1
slot         2    3  |  0    1
```

由此可见，逻辑上连续的位置 0、1、2、3，对应的物理行号依次为 2、3、0、1。对模型而言历史仍是连续的，对物理页池而言各页不必相邻。

运行时通常不是逐点调用，而是对一段逻辑区间批量计算。给定区间的起止位置后，对其中每个逻辑下标应用同一公式，得到与区间等长的一维整型向量。该向量中第 i 个元素，即本步第 i 个新 token 应写入的 slot。单点翻译与区间翻译遵循同一组规则。


## Store 按地址写入并按页表读回
> 源码见 [src/pagedattention/store.py](../src/pagedattention/store.py)

slots 解决了地址翻译，但仍缺少对页池的实际访问。
本步新算出的 K 与 V 需要按 slot 写入物理张量。
Attention 计算之前，又需要把已经写入的历史按逻辑顺序读出。
本仓库将这两种操作封装为 PagedStore。
该对象在构造时绑定一个 BlockPool，此后所有读写都作用于该池中的 k 与 v。
PagedStore 不负责订页，也不决定本步应覆盖哪一段逻辑区间。它只接受调用方给出的参数并完成访存。

先讨论**写入**。调用方提供本步的 k、v，以及与之等长的 slot 向量。实现上将 k、v 规范为每个 token 占据一行的布局，并将指定层的池子张量展成二维表，使行号与 slot 一一对应。随后执行按行拷贝，使第 i 个新 token 的内容落到第 slot_i 行。由于 slot 可以指向互不相邻的物理行，写入结果在显存中不必连续。本仓库使用框架提供的索引写入完成上述分散存储，而不另行实现专用 CUDA 核。其语义是按已翻译好的物理行号完成散射写。（后续会有优化的空间）

再讨论**读回**。Attention 需要的是从逻辑位置 0 到当前序列长度的一段有序 K 与 V。这些数据在页池中可能分布在不同物理块上。读回时，调用方给出页表以及待读取的逻辑长度。对每一个逻辑位置，采用与 slots 相同的整除与取模规则，得到物理块号与页内偏移，再通过高级索引一次性取出对应元素，并按逻辑顺序拼接。于是写入路径以 slot 向量为索引，读回路径以页表与长度为索引。二者依据同一套虚实地址关系，只是索引形式不同。

PagedStore 的职责止于访存。谁准备 slot，谁准备页表，读多长，写哪一层，均由调用方决定。该模块既不保存请求级状态，也不区分 Prefill 与 Decode。

教学实现里，gather 会先把分散在各页上的历史收成一段连续张量，再交给普通 SDPA。这与工业界「kernel 内按页打分」不同，但语义等价，便于先把页表关系讲清楚。




## Context 本步前向的访存契约
> 源码见 [src/kv/context.py](../src/kv/context.py)

至此，存放与寻址所需的模块已经齐备。
- Sequence 记录请求的逻辑状态，
- BlockPool 提供物理空间，
- slots 完成地址翻译，
- PagedStore 提供按 slot 写入与按页表读回。


```py
@dataclass
class ForwardContext:
    """ Attention 只用页表合同。 """

    is_prefill: bool              # 是否 prefill 步
    cache_len: int                # 本步开始前已有 KV 长度
    seq_len: int                  # 本步写完后的总 KV 长度（= gather 的长度）
    slot_mapping: torch.Tensor    # 本步要写入的物理 slot
    block_table: list[int]        # 逻辑页 → 物理块号
    store: PagedStore             # 物理读写模块


```
要把这些能力接到 Attention，还必须回答一个计算问题。Attention 在每一步都需要对全部历史位置的 Key 与 Value 做加权。若没有缓存，历史只能在当步重算。引入缓存之后，历史可以先写入页池，再在后续步骤中读回。

Prefill 与 Decode 使用同一套页池与同一套读写接口，差别在于本步新写入的长度以及读回的长度。Prefill 通常一次处理较长的新 token 段。该阶段既要把新位置的 K 与 V 写入页池，又要基于已写入的历史完成注意力，从而得到供采样使用的表示。Decode 通常每次只处理一个新 token。该阶段只需计算当前位置的 Q、K、V，将新的 K 与 V 追加写入，再读出截至当前的全部历史，与长度为 1 的查询做注意力。

模块功能清楚之后，仍缺少计算层与资源层之间的对接方式。Attention 的每一层若在 forward 中直接查询管理器，或把整份 Sequence 沿网络向下传递，则计算依赖会与生命周期管理缠结在一起。因此需要一个尽量窄的接口。该接口只携带本步读写缓存所必需的信息，在一次前向开始前安装，供各层以统一方式读取。本仓库将该接口实现为 `ForwardContext`。

ForwardContext 不是跨请求的长期状态，而只是单次前向有效的临时契约。其内容由六个字段组成，并通过 set_forward_context 与 get_forward_context 安装或清除。调用模型之前安装契约。模型内部各层通过 getter 读取。本步或整次生成结束时再清除，以免后续请求误用上一份数据。

- cache_len 表示本步开始前，逻辑上已有多少 token 的 KV 位于页池中。
- seq_len 表示本步写完后的总长度，也即 gather 应读取的长度。二者之差等于本步新 token 的个数。该差值必须与 slot_mapping 的长度相等，也必须与模型本步输入序列的长度相等。任一不等都表明契约填写错误。
- slot_mapping 给出本步待写入的物理行号。
- block_table 给出整段请求的页表，供读回历史使用。
- store 指向绑定在页池上的 PagedStore。
- is_prefill 为阶段标记。实际分支更多地依据长度关系，而不是单独依赖该标记。

因此 Context 的作用可以概括如下。它把本步所需的 store、页表、写入槽位与读回长度收成计算层唯一依赖的输入。


## Attention 消费契约并完成注意力
> 源码见 [src/layers/attention.py](../src/layers/attention.py)

Context 给出了本步访存所需的全部参数。Attention 的职责是在 forward 中消费这份契约。先把 q、k、v 转到注意力布局，再读取契约。无契约时退化为当场因果注意力，不碰页池。有契约时固定走三步：write → gather → attend。写入只覆盖本步新增位置，读回覆盖截至当前的全部历史。

```py
def forward(self, q, k, v):
    ctx = get_forward_context()
    q = q.transpose(1, 2)  # [B, H, S, D]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if ctx is None:  # 无缓存路径
        out = self._sdpa(q, k, v, is_causal=True)
        return out.transpose(1, 2).contiguous()

    start, n, total = ctx.cache_len, q.shape[2], ctx.seq_len
    assert total == start + n

    # 1) write：Post-RoPE 的 K/V 按 slot_mapping 写入页池（仅新增位置）
    ctx.store.write(self.layer_id, ctx.slot_mapping, k.transpose(1, 2), v.transpose(1, 2))

    # 2) gather：按 block_table 读回逻辑长度 seq_len 的全部历史
    k_c, v_c = ctx.store.gather(self.layer_id, ctx.block_table, total)
    k_c = k_c.unsqueeze(0).transpose(1, 2)  # [1, Hkv, T, D]
    v_c = v_c.unsqueeze(0).transpose(1, 2)

    # 3) attend：本步 Q × 读回的历史；掩码随阶段分支
    if start == 0 and n == total:          # 整段 Prefill：标准因果
        out = self._sdpa(q, k_c, v_c, is_causal=True)
    elif n == 1:                           # Decode：单查询，无需因果掩码
        out = self._sdpa(q, k_c, v_c, is_causal=False)
    else:                                  # 前缀命中后的部分 Prefill：显式掩码
        q_pos = torch.arange(start, start + n, device=q.device)[:, None]
        kv_pos = torch.arange(total, device=q.device)[None, :]
        attn_mask = torch.zeros(n, total, device=q.device, dtype=q.dtype)
        attn_mask = attn_mask.masked_fill(~(kv_pos <= q_pos), torch.finfo(q.dtype).min)
        out = self._sdpa(q, k_c, v_c, is_causal=False, attn_mask=attn_mask)

    return out.transpose(1, 2).contiguous()
```

第三步的掩码规则可以概括为三支：`cache_len == 0` 且本步长度等于总长度时用标准因果，对应从空缓存起步的整段 Prefill；本步查询长度为 1 时不再加因果掩码，对应典型 Decode；`cache_len > 0` 且查询长度大于 1 时显式构造掩码，使每个新查询只能看见不超过自身逻辑位置的历史，对应前缀命中后的部分 Prefill。

Attention 只依赖 ForwardContext。有契约则 write → gather → attend，无契约则当场计算。单请求内的缓存计算路径至此闭合。


## Manager 页的生命周期编排
> 源码见 [src/kv/manager.py](../src/kv/manager.py)

前面各模块各自都只负责一件事，Sequence 记逻辑状态，Pool 管物理页与三态，Store 管按地址读写，Prefix 管满块能否跨请求复用。还缺一个入口把它们串起来，并只对外暴露「订页 / 补页 / 还页」。这个入口就是 `KVManager`。


### 构造时组装什么

```py
self.pool = BlockPool.allocate(model, ...)   # 物理页池
self.store = PagedStore(self.pool)           # 给 Attention 的唯一读写口
self.index = _make_index(prefix_backend)     # none | hash | radix
self.pool.on_evict = lambda bid: self.index.invalidate(bid, self.pool)
```
Pool 出空间，Store 包一层访存，Index 按配置挂前缀策略。淘汰回调保证 cached 页被挤掉时，索引不会继续指向已失效的块。之后 Engine 只拿 Manager；Attention 只经 Context 拿到这里创建的 `store`。

### 四个对外动作

一条请求从进入引擎到离开，管理器在四个时刻被使用。对应源码里的四个方法。

**先创建**。`new_sequence` 根据 prompt 造出一条 Sequence。这时手里只有 token 列表和页大小，页表还是空的。它只登记「这条请求是谁」，并不分配任何物理页。

**再订页**。真正领页发生在 Prefill 之前，方法叫 `allocate`。管理器先询问前缀索引：当前 prompt 前面有没有已经算过、可以整页复用的满块。若有，就把这些物理页写进页表；页若还停在 cached，就拉回 used；若已被别的请求占用，就增加引用计数。前缀用完之后仍缺的页，一律向页池新领。最后根据命中了多少满块，写下本请求可以跳过的 token 数，供引擎决定 Prefill 从哪里开始喂模型。

这里有一个容量上的细节。命中很长并不总是好事：复用页占住池子之后，剩余空闲可能不够放下本请求尚未命中的那一段。因此 `allocate` 会在必要时主动缩短命中长度，宁可少复用一点，也要保证后面的新页能领到手。

```py
hit = self.index.match(seq, self.pool)
max_cached = min(len(hit), need)
while max_cached >= 0:
    new_needed = need - max_cached
    reclaimable = sum(1 for b in hit[:max_cached] if b in self.pool.cached_ids)
    # free + 即将 reclaim 的 cached，必须够放下未命中的新页
    if len(self.pool.free_ids) + reclaimable >= new_needed:
        break
    max_cached -= 1
if max_cached < 0:
    raise RuntimeError("not enough KV blocks for sequence")
```

这一段表示假设复用前 `max_cached` 页，还要新领 `new_needed` 页；命中里仍停在 cached 的页可以顺便拉回，算进可用容量。若仍不够，就少复用一页再试，直到装得下，或彻底不够而报错。

> 注意：订页只决定**数据将写在哪**。新领来的页此时往往仍是空的，数字要等 Attention 写入之后才有。

**然后补页**。Decode 每生成一个新 token，序列就可能跨过页边界。引擎在追加 token 之后调用 `ensure_blocks`。它只看一件事：当前长度需要几页，页表里够不够。不够就继续向池子领新页，追加到页表末尾。补页不查前缀索引，因为多出来的永远是本请求自己的尾部，别人无法共享。对应代码同样在 `src/kv/manager.py`：

```py
def ensure_blocks(self, seq: Sequence) -> None:
    while len(seq.block_table) < seq.num_blocks:
        seq.block_table.append(self.pool.allocate_fresh())
```

`seq.num_blocks` 由当前 token 数向上取整得到。页表短了就 `allocate_fresh`，直到长度对齐。

**最后还页**。请求结束时调用 `finish`。管理器先把已经填满的页登记进前缀索引，供以后相同前缀复用；未填满的尾页不能登记。随后逐页归还：已登记的满块在开启前缀时进入 cached，其余进入 free。页表字段清空，这条 Sequence 与物理页脱钩。若使用 radix 后端，还要额外放下该请求在树上的引用。对应代码：

```py
def finish(self, seq: Sequence) -> None:
    n_full = seq.num_full_blocks()
    if n_full > 0:
        self.index.publish(seq, n_full, self.pool)   # 只登记满块

    if isinstance(self.index, RadixPrefixIndex):
        self.index.release_seq(seq.seq_id)

    published = set(seq.block_table[:n_full]) if n_full > 0 else set()
    for bid in list(seq.block_table):
        keep = bid in published and self.prefix_backend != "none"
        if not keep and self.prefix_backend == "hash":
            self.index.invalidate(bid, self.pool)
        self.pool.release_to_cached_or_free(bid, keep_cached=keep)

    seq.block_table.clear()
    seq.num_cached_tokens = 0
```

要点是两段：先 `publish` 满块；再按 `keep` 决定进 cached 还是 free。尾页不在 `published` 里，因此不会留下给前缀复用。

把四次调用串起来，时间线就是：

```text
new_sequence          登记请求，页表为空
    ↓
allocate              Prefill 前订页，可能复用前缀
    ↓
（引擎绑 Context，Attention 写入并读回）
    ↓
ensure_blocks         Decode 变长时按需补页（可多次）
    ↓
finish                发布满块，页回 cached 或 free
```


## Engine 一次生成的时间线
> 源码见 [src/engine.py](../src/engine.py)

Manager 解决了页归谁；Attention 解决了有契约时怎么写、读、算。还缺一层把整次生成串起来。

运行时真正驱动缓存路径的，是 `_bind_ctx` 与 `generate`。

### 安装本步契约：`_bind_ctx`

每一步前向之前，引擎必须告诉各层 Attention：本步新写入从哪写到哪、历史读多长、页表是什么、store 在哪。这些信息收成 `ForwardContext`，由 `_bind_ctx` 安装：

```py
def _bind_ctx(self, seq, *, is_prefill, cache_len, write_end):
    slots = slots_tensor(
        seq.block_table, cache_len, write_end,
        self.kv.block_size, device=self.device,
    )
    set_forward_context(ForwardContext(
        is_prefill=is_prefill,
        cache_len=cache_len,
        seq_len=write_end,          # gather 读到这里
        slot_mapping=slots,         # 写入 [cache_len, write_end)
        block_table=seq.block_table,
        store=self.kv.store,
    ))
```

`cache_len` 是本步开始前页池里已有的逻辑长度；`write_end` 是写完后的总长度。二者之差等于本步新 token 个数，也等于 `slot_mapping` 的长度。装好之后，模型每一层通过 `get_forward_context` 读到同一份契约。

### 一次 `generate` 怎么走

入口先把 prompt 编成 `input_ids`，再交给管理器：

```py
seq = self.kv.new_sequence(input_ids)
self.kv.allocate(seq)
cached = seq.num_cached_tokens
prompt_len = len(input_ids)
```

订页完成后，按前缀命中情况分成两支，目的都是得到第一个新 token 的 logits。

**普通 Prefill**（`cached < prompt_len`）：只把未缓存后缀送进模型，契约覆盖区间 `[cached, prompt_len)`。

```py
self._bind_ctx(seq, is_prefill=True, cache_len=cached, write_end=prompt_len)
ids = torch.tensor([input_ids[cached:]], ...)
positions = torch.arange(cached, prompt_len, ...).unsqueeze(0)
h = self.model(ids, positions)
logits = self.model.compute_logits(h[:, -1:]).squeeze(1)
```

**整段前缀命中**（`cached == prompt_len`）：页里已有全部 prompt 的 K/V，但仍需一次长度为 1 的前向才能取出 logits 做采样。契约写成只覆盖最后一个位置：

```py
self._bind_ctx(seq, is_prefill=False, cache_len=prompt_len - 1, write_end=prompt_len)
# 只喂 input_ids[-1]，再采样
```

采出首个新 token 后，将其追加进 Sequence，并 `ensure_blocks` 补页。随后进入 Decode 循环：每步输入长度为 1，契约为 `[cache_len, cache_len + 1)`，前向、采样、追加、补页，直到 EOS 或达到长度上限。

```py
self._bind_ctx(seq, is_prefill=False, cache_len=cache_len, write_end=cache_len + 1)
ids = torch.tensor([[next_id]], ...)
positions = torch.tensor([[cache_len]], ...)
# forward → sample → append_token → ensure_blocks
```

整次生成包在 `try/finally` 里。无论正常结束还是中途返回，都会清掉契约并还页，避免下一趟请求误用上一份 Context，或页泄漏在 used 里：

```py
finally:
    set_forward_context(None)
    self.kv.finish(seq)
```

把引擎视角的时间线压缩如下：

```text
encode prompt
    → new_sequence / allocate
    → Prefill（或全命中时的单 token 前向）→ sample
    → append / ensure_blocks
    → Decode 循环：bind → forward → sample → append / ensure_blocks
    → finally：清 Context + finish
```



### 示例

设定 `block_size = 4`，prompt 有 10 个 token，无前缀命中，再生成若干新 token。

**allocate**：需要 `ceil(10/4) = 3` 页，例如页表 `[2, 5, 0]`，`num_cached_tokens = 0`。

**Prefill**：`cache_len=0, write_end=10`，模型输入长度 10；write 覆盖逻辑 0..9，gather 长度 10；采样得到首个生成 token `t0`。随后 `append(t0)`，长度变为 11，`ensure_blocks` 此时往往仍是 3 页。

**Decode 第 1 步**：`L=10`，契约为 `cache_len=10, write_end=11`，只喂 `t0` 与位置 10；write 一个 slot，gather 长度 11，再采样。

**继续 Decode**：长度每加 1，契约向前推一格。当长度到 13 时，`ceil(13/4)=4`，`ensure_blocks` 多领第 4 页，页表变长，再 bind 下一步的单 slot。

**finish**：若最终长度为 13，则 `num_full_blocks = 13//4 = 3`；前 3 页可 publish 进 cached，第 4 页未满，回 free。


## Prefix 跨请求的满块复用
> 源码见 [src/kv/prefix/](../src/kv/prefix/) · 细读见 [Prefix 详解](./03-Prefix详解.md)

同一次 `generate` 内，Decode 已不必重算历史。前缀是叠在页池上的第二层：跨请求复用满块，少做 Prefill。它主要挂在 `allocate`（`match`）与 `finish`（`publish`），不改 Attention 的 write/gather。`none` / `hash` / `radix` 的差别、以及同一组 token 下两种后端如何命中，见下一篇。

本教学 v1 **有意保持简单**：单请求 `generate`，gather 成 dense 再 SDPA，同步单卡。后续优化版本我们会逐步优化现在存在的性能瓶颈和批处理方案，以及更加丰富的前缀和淘汰协议，使其效果更好。

建议按下列顺序阅读源码。
| 顺序 | 文件 | 自检问题 |
| ---: | --- | --- |
| 1 | `sequence.py` | 一条请求的逻辑状态包含哪些字段 |
| 2 | `pool.py` | 物理 K 与 V 的张量形状是什么；页有哪三种状态 |
| 3 | `slots.py` | 逻辑位置如何变成 slot |
| 4 | `store.py` | write 如何使用 slot_mapping；gather 为何需要 block_table 与 seq_len |
| 5 | `context.py` | cache_len 与 seq_len 的差额表示什么；谁安装契约，谁读取契约 |
| 6 | `attention.py` 的 `forward` | 有契约时的三步是什么 |
| 7 | `manager.py` 的 allocate / ensure_blocks / finish | 它们分别对应请求生命周期的哪一时刻 |
| 8 | `engine.py` 的 `_bind_ctx` 与 `generate` | Prefill 与 Decode 各自传入的 cache_len 与 write_end 是什么 |
| 9（可选） | `hash_index.py`  | match 与 publish 如何接到 allocate 与 finish |


