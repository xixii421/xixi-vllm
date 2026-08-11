# 10 · Prefill CUDA Graph：piece CG
> !!!AI 辅助，暂未修改

> 对应 [04-CUDA-Graph](./04-CUDA-Graph.md)（decode CG）。这一份专门讲 **prefill 怎么用 CG**。
> 核心矛盾：prefill 的序列长度是动态的，CUDA Graph 却要求固定形状。
> v3 用 **piece CG** 解决首 chunk 的短 prompt 场景。

## 一、为什么 prefill 也能用 CG？

### 1.1 prefill 的瓶颈是 CPU 发射地板

eager prefill 单步 ~20ms，其中 GPU 计算只需 ~2.7ms，其余 ~17ms 是 **360 次 kernel 发射的 CPU 串行化**（GPU 在等 CPU）。这正是 CG 最擅长消除的开销。

但注意：这只对 **CPU-bound 区间**成立。长文本 prefill（>1536 token）是 GPU-bound（GPU 打满），CG 消除发射开销的收益趋近于零——这是 piece CG 的适用边界所在。

### 1.2 难点：动态长度

decode 每步恒为 bs 个 token（形状固定），CG 直接可用。prefill 的 token 数随 prompt 长度变化——**动态形状无法直接捕获**。

### 1.3 v3 的解法：按形状分片 + 有界覆盖

```
piece CG:  把 (batch_size, 长度) 空间切成离散的 (bs, L) 网格，
           只捕获网格上的固定形状图。请求落在网格上 → 重放；落空 → eager。
```

> 早期版本还有 **Continuation CG**（前缀命中续 chunk 的 q_len 网格图），因其
> `len(seqs)==1` 限制在并发前缀命中场景几乎不命中（实测 64 次前缀命中仅 4 次走
> cont CG），收益/成本比差（+2.5~2.7 GiB graph pool），已从 v3 移除。前缀命中
> 续 chunk 现在直接走 eager——radix 已跳过 KV 重算，剩余待算 token 较少。

---

## 二、Piece CG（`core/prefill_cuda_graph.py`）

### 2.1 piece 网格设计

```python
# bs=1 用 1.5x 中点网格，把「pad 到下一个 2 的幂」的最坏浪费压到 ~1.25x
fine_L = (128, 192, 256, 384, 512, 576, 640, 704, 768, 896, 1024, 1152, 1280, 1536)
# bs≥2 按 bs 定制（bs*L ≤ cg_max_tokens=1536）
per_bs_L = {2: (...), 4: (...), 8: (...)}
```

约束 `bs*L ≤ cg_max_tokens`（默认 1536）双重作用：
1. **限制捕获显存**：每个 piece 一个图，图越多 graph pool 越大。
2. **限制收益区间**：>1536 token 的长 prefill 已 GPU-bound，padded CG 反而不如 eager，无需捕获。

### 2.2 padding 语义（数值正确性的关键）

piece 捕获固定 L，但真实请求长度可能 < L。真实 token 放 `[i*L, i*L+len_i)`，padding 行保持 0/`-1`：

```python
# replay() 里
in_ids.zero_(); pos.zero_(); slot.fill_(-1)
for i in range(bs):
    n = seq_lens[i]
    in_ids[i*L : i*L+n].copy_(input_ids[off:off+n])   # 真实 token
    # padding 行：pos=0, slot=-1，FA2 对 position 0 的输入天然 causal 排除
```

piece CG 内走 **FA2 varlen**（连续 KV 直算，比 FlashInfer paged prefill 快），`cu_seqlens_k` 与 `cu_seqlens_q` 同为 token 粒度——**这是 bs>1 数值正确的关键修复**（早期 bug：误用页粒度的 `kv_indptr` 当 `cu_seqlens_k`，导致每条 K 序列被当成 1 个 token，跨序列污染）。

### 2.3 收益判据（`can_use`）

```python
def can_use(self, bs, seq_lens):
    piece = 覆盖批次的最小 (bs, L)       # 贪心找
    if piece_gpu_ms(piece) < eager_floor_ms:   # CG 有益 ⇔ 比发射地板快
        return piece
    return None   # GPU-bound 区间主动放弃 CG
```

`_calibrate()` 用 n=512 的 eager forward wall 估 `eager_floor_ms`（CPU 发射地板）；`_measure_piece_replay()` 用 CUDA event 实测每个 piece 的纯 GPU 时间。**"CG 有益 ⇔ piece_gpu < S"** 自动吸收了不同模型规模/形状的 GEMM 效率差异（0.6B 上 ≤1500 token 有效，4B 上只有 ≤300 token 有效）。

### 2.4 静态 buffer 与 workspace 共享

所有输入输出预分配（`input_ids`/`positions`/`slot_mapping`/`qo_indptr`/`kv_indptr`/`kv_indices`/`kv_last_page`/`outputs`），重放时 `copy_` 覆盖。**FlashInfer workspace 全 runner 共享一块**（pieces 时序互斥），见 [09](./09-显存优化.md)。大模型 workspace 溢出时自动加倍重建（`_prepare_piece`）。

### 2.5 适用边界

| 场景 | piece CG | 原因 |
| --- | --- | --- |
| 短 prompt 独立请求（≤1536 token） | ✅ | 命中网格，6× TTFT 收益 |
| 长文本（>1536 token） | ❌ 回退 eager | 网格外 + GPU-bound 判据拒绝 |
| 前缀命中续 chunk | ❌ | 不是首 chunk（走 eager，radix 已跳过 KV 重算） |
| 长文本并发连续对话 | ❌ | 首轮并发 prefill 远超网格覆盖墙 |

> **真实场景结论**：piece CG 的收益集中在「短 prompt 独立请求」——聊天 API 的最常见形态，
> eager TTFT ~20ms → CG 3~7ms（5-7×）。长文本连续对话场景它几乎不命中，此时
> decode CG（每步恒定收益）与 radix 前缀缓存（跳过 KV 重算）才是主力。

---

## 三、三个 CG runner 的分工

| runner | 阶段 | 形状策略 | KV 更新 | workspace |
| --- | --- | --- | --- | --- |
| `cuda_graph.py` | decode | bs 网格 | replan（CSR 重组） | 私有 |
| `prefill_cuda_graph.py` | prefill 首 chunk | (bs, L) 网格 + FA2 | 静态（dummy page） | **共享** |

> 早期 `continuation_cuda_graph.py`（prefill 续 chunk，q_len 网格 + replan paged）已移除。

**知识点总结**：
1. CUDA Graph = 预分配静态 buffer + 覆盖数据 + 一次 replay，消除的是发射开销不是计算。
2. 动态形状处理有两种：分片网格（piece）、钳制到有限域（cont，已移除）。
3. "CG 有益"要实测判据，不是形状命中就用。
4. workspace 类资源在时序互斥时共享（省显存），graph pool 因中间形状不同近似 sum（调网格权衡）。
