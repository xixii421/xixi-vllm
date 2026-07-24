# vllm-v1 KV Management

之前的 v0 版本实际上实现了 vllm 的核心功能，包括：

- 实现最基本的自回归推理流程；
- 支持单请求、全序列重算；
- 实现完整的Qwen3架构
- 提供可选的采样策略；

v0 的重点在于如何复现最小可运行的 Qwen3 推理流程。具体可以参考 [vllm-v0/README.md](../vllm-v0/README.md) 获取详细介绍。

很多人一提 vllm，脑子里就闪现出 KV cache、continuous batching，没错，这些就是大家都关心、也公认 vllm 最具特色的部分。

V1版本就带着大家去实现 KV 管理，我们在 V0 的基础上进一步扩展。
下面是我做的一个版本迭代：

1. 物理 block 池
2. PagedAttention 教学模块
3. hash / radix 前缀缓存

> 非 KV 代码与 [vllm-v0](../vllm-v0/) 对齐。v0 的文件清单见 [docs/ADDED_vs_v0.md](docs/ADDED_vs_v0.md)。

## 模块一览


| 路径                        | 相对 v0  | 作用                                 |
| ------------------------- | ------ | ---------------------------------- |
| `src/pagedattention/`     | **新增** | 页表映射 + `PagedStore` 读写             |
| `src/kv/`                 | **新增** | 块分配 / 前缀索引 / ForwardContext        |
| `src/kv/pool.py`          | **新增** | 物理 block 池                         |
| `src/kv/prefix/*`         | **新增** | hash / radix / noop 前缀索引           |
| `src/engine.py`           | **改写** | prefill + decode；组装 `slot_mapping` |
| `src/layers/attention.py` | **改写** | `store.write` → `gather` → SDPA    |
| 其余 `src/*`                | 同步 v0  | 模型 / 采样 / loader                   |




## 快速开始

```bash
source .venv/bin/activate

cd vllm-v1

# 单次推理
python run.py --model Qwen3-0.6B --prompt "Hello" --max-new-tokens 64 --temperature 0 --prefix-backend hash

# 前缀命中测试
python scripts/check_prefix_hit.py --model Qwen3-0.6B --backend hash
python scripts/check_prefix_hit.py --model Qwen3-0.6B --backend radix

python test/pagedattention.py
# 端到端：三后端 greedy 一致 + 二次前缀命中（模型只加载一次）
python test/e2e.py --model Qwen3-0.6B

# 相对 v0 的核心性能对比（Decode 扩展 / Prefill 扩展 / 前缀二次命中）
python -m bench.compare_v0 --model Qwen3-0.6B
```

> `--prefix-backend`：`none`| `hash` | `radix`


## 和V0的性能对比

v1 的目标是把 **KV 管理 / 分页 / 前缀** 这条机制链路跑通，并保持代码可读。当前 Attention 仍是教学路径：


目前版本与 vLLM **语义接近，但是性能不等价**。v1 暂时比 v0 慢是预期现象，后续我也会告诉大家应该如何优化。

### 一组实测（Qwen3-0.6B，eager，warmup=1，repeats=3）

机器上自行复现：`python -m bench.compare_v0 --model Qwen3-0.6B`

**A) Decode 扩展**（短 prompt ≈30 token，增大 `max_new`）

| case | v0 E2E | v1/none E2E | E2E×（>1 表示 v1 更快） | v0 Dec t/s | v1 Dec t/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| new=32 | 1.12s | 3.91s | 0.29× | 28.7 | 8.3 |
| new=64 | 2.43s | 7.36s | 0.33× | 26.4 | 8.7 |
| new=128 | 4.88s | 14.86s | 0.33× | 26.2 | 8.6 |

**B) Prefill 扩展**（增大 prompt，`max_new=8`，无前缀命中）

| case | v0 TTFT | v1 TTFT | TTFT× |
| --- | ---: | ---: | ---: |
| p≈70 | 0.034s | 0.047s | 0.72× |
| p≈250 | 0.038s | 0.164s | 0.23× |

**C) 前缀二次命中**（同一 Engine，`hash`，长 prompt ≈130 token）

| | cached | TTFT | E2E |
| --- | ---: | ---: | ---: |
| 第 1 次 | 0 | 0.079s | 2.57s |
| 第 2 次 | 128 | 0.164s | 1.95s |

说明：第二次 `cached` 已上去、E2E 有所下降，说明 **前缀复用机制是通的**。本次 TTFT 未下降，主要仍被「后缀 Prefill + 后续 Decode 的 gather」拖住，属于同一类开销。

当前教学 v1 更慢。主因是每层每步都要把历史从页里 **gather 成连续张量**，长度越长拷贝越重。  


## 设计文档目录

入口：[docs/README.md](docs/README.md)

> [V1版本设计解释](./docs/00-V1版本设计解释.md) |
> [KV Cache代码详解](./docs/01-KV%20Cache代码架构解释.md) | 
> [Prefix详解](./docs/03-Prefix详解.md)