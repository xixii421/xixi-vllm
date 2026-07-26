# vllm-v1 — KV Cache + PagedAttention + 前缀缓存

> 在 v0 基础上实现 KV 管理：物理 block 池、PagedAttention 读写、hash/radix 前缀索引。

---

## 相对 v0 新增

| 新增能力 | 对应模块 | 说明 |
|---------|---------|------|
| 物理 Block 池 | `src/kv/pool.py` | 固定大小 block 分配/回收 |
| PagedAttention | `src/pagedattention/` | 页表映射 + `PagedStore` 读写（write/gather） |
| 前缀缓存 | `src/kv/prefix/` | hash 链式 / radix 树 / noop 三种后端 |
| ForwardContext | `src/kv/context.py` | 全局 `set/get_forward_context` 传递 KV 上下文 |
| prefill/decode 分离 | `src/engine.py` | prefill 读全量 KV + causal mask，decode 逐 token 增量 |
| slot_mapping | engine 侧构建 | 决定每个 token 写入哪个物理 slot |

**限制：** 仍是单请求模式，每步 gather 历史 KV 为连续张量再 SDPA，性能低于 v0（预期中）。

---

## 快速开始

```bash
cd vllm-v1

# 单次推理
python run.py --model Qwen3-0.6B --prompt "Hello"

# 切换前缀后端
python run.py --prefix-backend radix

# 前缀命中测试
python scripts/check_prefix_hit.py --model Qwen3-0.6B --backend hash

# 端到端测试
python test/e2e.py --model Qwen3-0.6B

# 相对 v0 性能对比
python -m bench.compare_v0 --model Qwen3-0.6B
```

---

## 核心源码

| 文件 | v0→v1 |
|------|-------|
| `src/kv/pool.py` | **新增** — 物理 block 池 |
| `src/kv/context.py` | **新增** — 全局 ForwardContext |
| `src/kv/prefix/hash.py` | **新增** — 链式 hash 前缀索引 |
| `src/kv/prefix/radix.py` | **新增** — 基数树前缀索引 |
| `src/pagedattention/store.py` | **新增** — PagedStore write / gather |
| `src/pagedattention/slots.py` | **新增** — slot_mapping 构建 |
| `src/engine.py` | **改写** — prefill + decode 双路径，组装 slot_mapping |
| `src/layers/attention.py` | **改写** — `PagedStore.write → gather → SDPA` |

---

## 文档

| 文档 | 说明 |
|------|------|
| [00-V1版本设计解释.md](docs/00-V1版本设计解释.md) | 为什么要做 KV Cache 和 PagedAttention |
| [01-KV Cache代码架构解释.md](docs/01-KV%20Cache代码架构解释.md) | Block 池 / PagedStore / slot_mapping 详解 |
| [02-相比v0的增改.md](docs/02-相比v0的增改.md) | v0 → v1 逐个差异文件清单 |
| [03-Prefix详解.md](docs/03-Prefix详解.md) | hash / radix 前缀索引实现 |
