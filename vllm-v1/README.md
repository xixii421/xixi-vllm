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

| 路径 | 相对 v0 | 作用 |
| --- | --- | --- |
| `src/pagedattention/` | **新增** | 页表映射 + `PagedStore` 读写 |
| `src/kv/` | **新增** | 块分配 / 前缀索引 / ForwardContext |
| `src/kv/pool.py` | **新增** | 物理 block 池 |
| `src/kv/prefix/*` | **新增** | hash / radix / noop 前缀索引 |
| `src/engine.py` | **改写** | prefill + decode；组装 `slot_mapping` |
| `src/layers/attention.py` | **改写** | `store.write` → `gather` → SDPA |
| 其余 `src/*` | 同步 v0 | 模型 / 采样 / loader |

## 快速开始

```bash
source .venv/bin/activate

cd vllm-v1

# 单次推理（打印 cached_prompt_tokens；首次几乎总是 0）
python run.py --model Qwen3-0.6B --prompt "Hello" --max-new-tokens 64 --temperature 0 --prefix-backend hash

# 前缀命中测试
python scripts/check_prefix_hit.py --model Qwen3-0.6B --backend hash
python scripts/check_prefix_hit.py --model Qwen3-0.6B --backend radix

python test/test_pagedattention.py
python test/test_e2e_smoke.py --model Qwen3-0.6B
```

> `--prefix-backend`：`none`| `hash` | `radix` 


## 设计文档目录

| 文档 | 内容 |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | **主讲义**：KV 核心、完整 generate 流程、与 Engine/Model 如何融合 |
| [docs/kv_cache.md](docs/kv_cache.md) | 分包、前缀对照、短数据流 |
| [docs/ADDED_vs_v0.md](docs/ADDED_vs_v0.md) | 相对 v0 增改清单 |

建议先读 `architecture.md`，再按其中第六节的路径跟源码。

