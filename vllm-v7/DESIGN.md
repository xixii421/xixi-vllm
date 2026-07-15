# vllm-v7 Design

主题：在 v6（TP/PP + 多模态 + CUDA Graph）之上对齐 **ullm（μLLM）** 的控制面，并为多机 / PD 分离做架构铺垫。

## ZMQ 是什么？和多进程 Pipe 差在哪？

| | `multiprocessing.Pipe` (v6) | **ZMQ** (v7 / ullm) |
|--|--|--|
| 形态 | OS 管道，一对一 | 消息队列套接字（PUB/SUB、PUSH/PULL、DEALER/ROUTER…） |
| 拓扑 | 主进程 × N 个 Pipe | 一条 **广播** + 一条 **driver 回程** |
| 跨机 | 不方便 | 原生 `tcp://` / `ipc://` |
| 序列化 | pickle 直接传 Python 对象 | 仍可 pickle；ullm 用 msgspec + 零拷贝 |

**NCCL 仍然只负责 GPU 张量 collective**（TP all-reduce、PP send/recv）。  
ZMQ / Pipe 只传 **控制消息**：`execute_model(seqs, …)`、`shutdown`、block 数量等。

```text
            ┌─────────────┐  ZMQ PUSH×N (per-rank cmd)  ┌──────────┐
 API/Async ─►│ Engine/主进程 │ ──────────────────────────►│ Worker 0… │──► NCCL
            │ Scheduler   │ ◄── ZMQ PUSH(all→engine)   │  TP/PP   │
            └─────────────┘                              └──────────┘
```

ullm 用 **PUB/SUB** 广播；v7 教学实现改用 **每 rank 一条 PUSH→PULL**，避免 PUB 慢订阅丢首包挂死。语义相同：控制面消息扇出 + driver（或全员）回程。

v7：`rpc_backend=zmq`（默认）或 `pipe`（兼容 v6）。

## ullm 如何设计？我们抄了什么 / 没抄什么

**ullm 有：**

- Engine 独立进程 + `EngineClient`（ZMQ）
- Worker PUB/SUB 广播 + driver PUSH 回结果
- Radix 前缀缓存（token→KV slot）
- Async scheduling（placeholder token）
- OpenAI FastAPI（`/v1/chat/completions`）
- TP / PP（Megatron 风格）

**ullm 没有：**

- 多模态（ViT）
- **PD 分离**（Prefill / Decode 异机）
- vLLM 完整 continuous batching 的全部调度策略

**v7 对齐点：**

| 能力 | v7 状态 |
|--|--|
| ZMQ worker RPC | ✅ `src/runtime/executor.py` + `zmq_rpc.py` |
| Pipe 回退 | ✅ `rpc_backend=pipe` |
| Radix 前缀缓存 | ✅ 按 **block hash** 建树（适配 FlashInfer page=256） |
| AsyncLLM + OpenAI API | ✅ `src/runtime/async_llm.py` + `entrypoints/openai_api.py` |
| 多机 NCCL | ✅ `nccl_host` / `nccl_init_method=tcp://host:port` |
| PD 分离 | 🟨 stub：`src/pd/`（接口 + 配置；传输未实现） |
| Async scheduling 深耦合 | 🟨 配置位 + 文档；调度重叠可继续加深 |

## Radix vs v6 hash 前缀缓存

- **v6**：`hash(block_tokens) → block_id` 扁平表  
- **v7 Radix**：`[h0,h1,…,hk] → [bid0,…,bidk]` 前缀树 + `inc_ref/dec_ref` + LRU `evict`  

单位是 **完整 KV page（256 tokens）**，不是 ullm 的 per-token slot——因为本仓库 KV 是 paged FlashInfer。

关掉：`EngineConfig(enable_radix=False)` 或 API `--no-radix`。

## TP / PP 在多机多卡上怎么跑

### 单机多卡（已有）

```bash
# TP=2
python run.py --model ~/huggingface/Qwen3-0.6B --tp-size 2 --device-ids 0,1 --rpc-backend zmq

# PP=2
python run.py --model ~/huggingface/Qwen3-0.6B --pp-size 2 --device-ids 0,1 --rpc-backend zmq
```

Rank 布局（与 v6/Megatron 一致）：

```text
rank = pp_rank * tp_size + tp_rank
device_ids[rank] → 物理 GPU
```

### 多机多卡（v7）

1. **每台机器各起 world_size 中属于本机的 worker**（教学版简化：仍由一台机 spawn 全部进程时，要求 RDMA/网络 CUDA 可见——真实生产应每节点 spawn 本地 rank）。  
2. NCCL：`tcp://MASTER_ADDR:MASTER_PORT`，所有 rank 可达。  
3. ZMQ：同机可用 `ipc://`；跨机把 broadcast/response 改成 `tcp://`（扩展点在 `zmq_rpc.tcp_path`）。

当前教学路径（控制面仍本地 spawn，数据面可指向远程 NCCL master）：

```bash
# 节点 0（master）
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.
python run.py --model ~/huggingface/Qwen3-0.6B \
  --tp-size 2 --pp-size 1 --device-ids 0,1 \
  --nccl-host 192.168.1.10 --nccl-port 29500 --rpc-backend zmq
```

**真正多机**需要：每个节点只启动本机的 local ranks，并用 `nnodes/node_rank` 算 global rank（见 `scripts/multinode_notes.md`）。PD 同理。

### 异构卡注意（你的第二台：PRO6000 + A100 + 4×3090）

| 组合 | 建议 |
|--|--|
| 单卡吞吐 | PRO6000 或 A100 单独跑最大模型 |
| TP | **只在同构卡上做**（4×3090 互 TP）；异构 TP 易踩 NCCL/能力差异 |
| PP | 可跨异构（各 stage 算子集合不同）；把 **重 stage（含 ViT）放大显存卡** |
| PD 未来 | Prefill→A100/PRO6000；Decode→3090 集群 |

验证顺序建议：

1. 单卡 smoke（每张卡各跑一次 `run.py --eager`）  
2. 4×3090：`--tp-size 2` / `--tp-size 4`（显存够）  
3. A100 + 一张 3090：`--pp-size 2 --device-ids <a100>,<3090>`  
4. OpenAI：`python -m src.entrypoints.openai_api --model … --port 8000`  
5. Radix：同一长 system prompt 连打两次，看第二次 TTFT  

## PD 分离（Prefill / Decode）应如何实现

```text
          HTTP / AsyncLLM
               │
     ┌─────────┴─────────┐
     ▼                   ▼
 Prefill Engine      Decode Engine
 (大 batch, 大显存)   (小 batch, CG)
     │  KV blocks + meta  │
     └──── transfer ──────┘
           (NCCL P2P / RDMA / Mooncake / shm)
```

v7 stub：

- `PDConfig.role ∈ {COLLOCATED, PREFILL, DECODE}`
- `KVTransferEngine` protocol + `NullKVTransfer`（直接 raise，防误开）

落地清单（下一迭代）：

1. Prefill 结束：导出 `block_table` + layer KV pages  
2. Transfer：按 layer 打包；TP 内 collective 后再发对端  
3. Decode：按 meta 重建 Sequence + block block 映射  
4. Scheduler：Prefill 队列 / Decode 队列分治  
5. 不要把 PD 与 PP 混为一谈——PP 是层切分；PD 是请求阶段切分  

## 异步调度 + OpenAI

- `AsyncLLM`：`asyncio.to_thread(Engine.step)`，锁串行保护调度器  
- `enable_async_scheduling`：与 ullm 一样 **禁止 PP**；深重叠仍可按 placeholder token 加强  
- API：`POST /v1/chat/completions`、`/v1/completions`，支持 `stream`

```bash
python -m src.entrypoints.openai_api \
  --model ~/huggingface/Qwen3-0.6B --port 8000 --eager
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'
```

## 目录增量（相对 v6）

```text
src/runtime/zmq_rpc.py      ZMQ helpers
src/runtime/executor.py     zmq | pipe
src/runtime/async_llm.py
src/scheduling/radix.py
src/scheduling/block_manager.py   enable_radix
src/pd/                     PD stubs
src/entrypoints/openai_api.py
scripts/multinode_notes.md
```
