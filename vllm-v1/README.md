# vllm-v1 — KV Cache + PagedAttention + 前缀缓存

> 在 v0 基础上实现 KV 管理：物理 block 池、PagedAttention 读写、hash/radix 前缀索引。

---

## 相对 v0 新增

| 新增能力 | 对应模块 | 说明 |
|---------|---------|------|
| 物理 Block 池 | `src/kv/pool.py` | 固定大小 block 分配/回收，free / used / cached 三态 |
| 序列逻辑状态 | `src/kv/sequence.py` | 一条请求的 token 序列 + 页表 + 前缀跳过长度 |
| KV 生命周期入口 | `src/kv/manager.py` | 订页（allocate）/ 补页（ensure_blocks）/ 还页（finish） |
| PagedAttention | `src/pagedattention/` | slot 映射 + `PagedStore` 读写（write 散射 / gather 聚合） |
| 前缀缓存 | `src/kv/prefix/` | 链式 hash 字典 / radix 压缩前缀树 / noop 三种后端 |
| ForwardContext | `src/kv/context.py` | 全局 `set/get_forward_context` 传递本步访存契约 |
| prefill/decode 分离 | `src/engine.py` | prefill 读全量 KV + causal mask，decode 逐 token 增量 |

**限制：** 仍是单请求模式，每步 gather 历史 KV 为连续张量再 SDPA，性能低于 v0（预期中）。

---

## 快速开始

```bash
cd vllm-v1

# 单次推理（使用 config.yaml 默认值，首次运行自动下载模型）
python run.py

# 指定模型 / prompt
python run.py --model Qwen3-1.7B --prompt "你好"

# 切换前缀后端（none / hash / radix）
python run.py --prefix-backend radix

# 管理命令
python run.py --list       # 查看可下载模型与显存预算
python run.py --download   # 预下载全部模型到 ~/huggingface
```

---

## 核心源码

新增（`src/` 目录下）：

| 模块 | 文件 | 职责 |
|------|------|------|
| KV 生命周期 | `src/kv/pool.py` | 物理 block 池，free / used / cached 三态 |
| | `src/kv/sequence.py` | 请求逻辑状态 + 页表 |
| | `src/kv/manager.py` | 订页（allocate）/ 补页（ensure_blocks）/ 还页（finish） |
| | `src/kv/context.py` | 全局 `ForwardContext` 访存契约 |
| 前缀缓存 | `src/kv/prefix/hash_index.py` | 链式 hash 字典索引 |
| | `src/kv/prefix/radix_index.py` | radix 前缀索引（match/publish/pin/release） |
| | `src/kv/prefix/radix_tree.py` | 压缩前缀树（insert / match_prefix / 引用计数） |
| | `src/kv/prefix/hash_util.py` + `noop.py` | 块哈希计算 / 空实现 |
| PagedAttention | `src/pagedattention/store.py` | PagedStore write（散射）/ gather（聚合） |
| | `src/pagedattention/slots.py` | 逻辑位置 → 物理 slot |

改写：

| 文件 | 变化 |
|------|------|
| `src/engine.py` | prefill + decode 双路径，组装 slot_mapping，驱动 KVManager |
| `src/layers/attention.py` | 走 `PagedStore.write → gather → SDPA` |

---

## 文档

| 文档 | 说明 |
|------|------|
| [00-设计解释.md](docs/00-设计解释.md) | 设计动机：v0 性能瓶颈、为何只有 KV 可缓存、PagedAttention 思想、KV 管理三理念 |
| [01-代码架构.md](docs/01-代码架构.md) | 代码详解：页池三态 / slot 映射 / PagedStore / KVManager 生命周期 / Engine 时间线 |
| [02-相比v0的增改.md](docs/02-相比v0的增改.md) | v0 → v1 差异文件清单（新增 / 继承 / 改写） |
| [03-Prefix详解.md](docs/03-Prefix详解.md) | 前缀缓存：链式 hash 字典 vs Radix 压缩前缀树，两种后端对比 |
