# vllm-v4 — 多机多卡并行（DP + TP + PP）

> 以 v3 为基底迁移，实现**数据并行 DP**、**张量并行 TP** 与**流水线并行 PP**，
> 控制面按 vLLM V1 的 `EngineCore / Executor / EngineCoreClient` 分层重构，补齐
> `torchrun` 多机启动与 NCCL 网络配置。

---

## 并行布局


| 维度  | 切分对象              | 通信                      | 布局             |
| --- | ----------------- | ----------------------- | -------------- |
| DP  | 请求副本              | ZMQ（门面↔EngineCoreProc） | 跨副本独立引擎       |
| TP  | head / 权重 / vocab | all-reduce / all-gather | 机内（NVLink 高带宽） |
| PP  | 隐藏层               | send / recv 中间张量        | 跨机（低带宽需求）      |


rank 约定：`global rank = pp_rank * tp_size + tp_rank`。TP 在组内连续，PP 跨组步进。

```
[node 0]                          [node 1]
Engine(Scheduler, rank0 兼 driver)  Worker(PP stage1, TP0)
  └─ Worker TP0 ◄─ all-reduce ─► Worker TP1
       │  PP 传 hidden+residual（跨机 TCP/RDMA）
       └──────────► Worker TP0 ──► Worker TP1
```

---



## 多机部署（代码同步 + 环境配置）

同一套代码跑多台机器，代码走 git、环境走脚本，不要手动拷文件。

```bash
git clone git@github.com:dzzy-team/dzyy-vllm.git
cd dzyy-vllm/vllm-v4
bash setup_env.sh          # 自动探测 CUDA 版本并装好环境
```

之后每次更新只需 `git pull` 和 `source .venv/bin/activate`。完整工作流见 `docs/多机部署.md`。

---

## 文档

并行相关的教材拆成几篇，入口是 `docs/并行环境.md`。

| 文档 | 内容 |
| --- | --- |
| [docs/并行环境.md](docs/并行环境.md) | 三类并行策略、代价、决策树 |
| [docs/硬件互联.md](docs/硬件互联.md) | 卡数、异构、NVLink 与 PCIe |
| [docs/服务器测量.md](docs/服务器测量.md) | 带宽测量、三台真实机器剖析 |
| [docs/集合通信.md](docs/集合通信.md) | NCCL 原语、进程组、控制面通信 |
| [docs/linear.md](docs/linear.md) | TP 线性层 |
| [docs/词表.md](docs/词表.md) | 词表并行 |
| [docs/多机部署.md](docs/多机部署.md) | git 同步与环境脚本 |

---

## 快速开始

```bash
cd vllm-v4
PY=/path/to/python          # 需 torch + flash_attn + flashinfer + transformers

# 单卡（tp=1 pp=1，与 v3 一致）
$PY run.py --model Qwen3-0.6B

# 单机多卡 TP（机内 NVLink，Engine 主进程 spawn 各卡 worker）
$PY run.py --model Qwen3-0.6B --tp 2 --pp 1

# 多机（torchrun，每卡一进程，rank0 兼调度器）
torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 \
    --master_addr=<node0_ip> --master_port=29500 run_torchrun.py --tp 2 --pp 2
torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 \
    --master_addr=<node0_ip> --master_port=29500 run_torchrun.py --tp 2 --pp 2
```

配置并行参数见 `config.yaml`：`tp_size / pp_size / nccl_port / master_addr / master_port / nccl_ifname`。

---



## 架构

收敛为「三个核心平面 + 一个分布式基座」，依赖单向无环：
`data`（纯 CPU）→ `compute`（依赖 data + distributed）→ `control`（依赖 data + compute + distributed）；
`distributed` 只依赖 torch。

| 文件 | 职责 |
| --- | --- |
| `src/control/engine.py` | `Engine`：用户门面（tokenizer / 输入输出 / 流式） |
| `src/control/core.py` | `EngineCore`：与传输/进程模型无关的纯控制循环（调度→前向→采样→更新） |
| `src/control/scheduler.py` | `Scheduler`：请求队列、prefill/decode 调度、KV 块分配 |
| `src/control/client.py` | `EngineCoreClient` 抽象 + `InprocClient`（dp=1） |
| `src/control/mp_client.py` | `MPClient`：dp>1 副本 spawn + ZMQ 路由 + 输出汇总（含 `run_engine_core_proc`） |
| `src/control/executor.py` | `Executor` 抽象 + `UniprocExecutor` / `MultiprocExecutor` + `create_executor` |
| `src/compute/worker.py` | 单 rank 运行时：`init_environment / initialize / execute_model / read_samples` |
| `src/compute/model_runner.py` | `ModelRunner`：组装 Batch、驱动模型、采样 |
| `src/compute/kv_pool.py` | `KVPool`：Worker 侧 GPU KV 张量 + `get_kv_dims` / `auto_num_blocks` |
| `src/compute/batch.py` | `Batch`：常驻 GPU 输入缓冲（prefill / decode / CUDA Graph 共用） |
| `src/compute/kv_ops.py` | Triton `store_kvcache` 内核 |
| `src/data/kv.py` | `KVManager`：driver 侧纯逻辑 KV 账本（块表 + 前缀索引，不占 GPU） |
| `src/data/sequence.py` / `batch.py` | `Sequence` 与 wire 结构（`ScheduledBatch / SchedulerOutput / EngineCoreOutput`） |
| `src/distributed/parallel_state.py` | `Group`（NCCL+Gloo 双组）、`WORLD/TP/PP`、rank 映射、`set_nccl_env` |
| `src/distributed/communication_op.py` | `all_gather/all_reduce/reduce_scatter`、TP 封装、PP `send/recv_tensors` |
| `src/distributed/utils.py` | `divide / get_pp_indices / set_pp_layer_ranges / IntermediateTensors / PPMissingLayer` |
| `src/distributed/control_channel.py` | 控制面 Gloo：广播调度、回收采样 counts + token ids |


- **控制面 = vLLM V1 形状**：`Engine`（门面）→ `EngineCoreClient` → `EngineCore`（纯循环）
  → `Executor`（`collective_rpc` 广播到各 worker）。
- **DP**：每个 `dp_size` 副本一个 `EngineCoreProc` 进程，各自持有一份独立引擎并发前向，
  门面通过 ZMQ `PUSH/PULL` 做 round-robin 路由与输出汇总（对齐 vLLM `MPClient`）。
- **TP×PP**：每副本内每卡一进程、rank0 兼 driver；单机 `run.py --tp/--pp` spawn 子进程，
  多机 `run_torchrun.py` 由 torchrun 拉起。控制面走 Gloo `ControlChannel`，计算面走 NCCL。
- **PP 是同步切层**，不是 1F1B 流水线。要装更大模型继续同步 PP + 不均分；要多卡吞吐用 DP。

---



## 正确性验证

```bash
# 单卡基线 / 单机 TP / PP（需对应数量空闲卡）
$PY test/test_parallel.py --tp 1 --pp 1
$PY test/test_parallel.py --tp 2 --pp 1
$PY test/test_parallel.py --tp 1 --pp 2
$PY test/test_parallel.py --tp 2 --pp 2   # 需 4 张卡

# 数据并行（dp_size 个独立引擎副本并发，需 >= dp_size 张卡）
$PY test/test_dp.py --dp 2 --n-reqs 4
```

`test_parallel.py` 以 **fp32 的 HF 参考输出**为 golden 断言逐 token 一致（而非 bf16
单卡）。实测（Qwen3-0.6B，4070 SUPER + 4090 D，16 token greedy）：


| 配置        | 结果                                                                |
| --------- | ----------------------------------------------------------------- |
| tp=1 pp=1 | 与 fp32 在 step1 的 40/785 近 tie 上因 bf16 取整而分叉（预有现象，非 v4 回归）         |
| tp=2 pp=1 | **PASS**，与 fp32 逐 token 一致                                        |
| tp=1 pp=2 | **PASS**，与 fp32 逐 token 一致                                        |
| tp=2 pp=2 | 需 4 卡，组件已分别验证（`test_sharding.py` TP 分片 / `test_pp_comm.py` PP 通信） |


> 说明：bf16 下 greedy 采样可能遇到两 token logits 恰好相等，单卡与多卡因 all-reduce
> 求和顺序不同会打破 tie 得到不同 token。fp32 参考才是唯一稳定真值；tp=2 / pp=2 的
> bf16 输出均与 fp32 逐 token 一致，证明并行切分与集合通信实现正确。

> 注意：NCCL 不允许两个 rank 复用同一张 GPU（`Duplicate GPU detected`），因此多卡
> 路径必须各 rank 独占一张卡。

---

### 大模型 PP 切分验证（8B / 14B）

`--model` 指定模型路径；`--pp-layer-counts` 指定**不均分 PP** 每 stage 层数（适配
异卡显存，如 12GB + 24GB）。pp=2 纯层切分（中间只 send/recv hidden+residual，无算术
重排），故 pp=1 与 pp=2 输出应 bit 级一致。

```bash
# 8B：单卡 4070(12G) 装不下，pp=2 各卡 ~7.8 GiB 一起跑
$PY test/test_parallel.py --model ~/huggingface/Qwen3-8B --tp 1 --pp 1     # 4090 基线
$PY test/test_parallel.py --model ~/huggingface/Qwen3-8B --tp 1 --pp 2     # 4070+4090

# 14B（29.4G，任何单卡都装不下）：不均分 [12,28]，4070 扛 12 层、4090 扛 28 层
$PY test/test_parallel.py --model <Qwen3-14B> --tp 1 --pp 2 --pp-layer-counts 12,28
```

实测（4070 SUPER 12GB + 4090 D 24GB，greedy decode）：

| 模型 | 配置 | 单卡权重 | 输出 |
| --- | --- | --- | --- |
| 8B | tp=1 pp=1（4090） | ~15.6 GiB | `[198, 32313, 11, 1077, 752, ...]` |
| 8B | tp=1 pp=2 | **7.80 GiB / 卡** | 与 pp=1 **bit 级一致** |
| 14B | tp=1 pp=2 [12,28] | 8.95 / 18.95 GiB | `[198, 32313, 11, 1077, 594, 1490, ...]` |
| 14B | tp=1 pp=2 [11,29] | 8.32 / 19.57 GiB | 与 [12,28] **逐 token 一致** |

结论：PP 在异卡机器上把「单卡装不下的模型」按显存比例切分，层切分点不影响最终输出
（[11,29] 与 [12,28] 一致），且 8B 的 pp=1/pp=2 输出 bit 级一致，证明 PP 通信与切分
实现正确。

---



## 性能对比

统一 `enforce_eager=True` 隔离 TP/PP 通信开销（`bench/bench_parallel.py`，Qwen3-0.6B，
prompt=256，greedy decode）：


| 配置        | TTFT (ms) | decode tok/s (b=1) | 吞吐 tok/s (b=8) |
| --------- | --------- | ------------------ | -------------- |
| tp=1 pp=1 | 20.79     | 62.2               | 457.4          |
| tp=2 pp=1 | 34.88     | 40.7               | 310.2          |
| tp=1 pp=2 | 50.03     | 20.5               | 158.0          |


0.6B 模型计算量极小，通信开销占主导，TP/PP 呈负加速比属预期——并行只在模型大到
「计算 >> 通信」时才划算。该测量量化了通信开销这一设计目标。