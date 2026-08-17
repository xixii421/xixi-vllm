# vllm-v4 — 多机多卡并行（TP + PP）

> 以 v3 为基底迁移，实现**张量并行 TP** 与**流水线并行 PP**，补齐 `torchrun` 多机启动与 NCCL 网络配置。

---

## 并行布局


| 维度  | 切分对象              | 通信                      | 布局             |
| --- | ----------------- | ----------------------- | -------------- |
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

沿用 v3 的「控制面 / 数据面 / 计算面」分层，新增一层 `distributed` 基座：


| 文件                                     | 职责                                                                       |
| -------------------------------------- | ------------------------------------------------------------------------ |
| `src/distributed/parallel_state.py`    | `Group`（NCCL+Gloo 双组）、`WORLD/TP/PP`、rank 映射、`set_nccl_env`               |
| `src/distributed/communication_op.py`  | `all_gather/all_reduce/reduce_scatter`、TP 封装、PP `send/recv_tensor_dict`  |
| `src/distributed/utils.py`             | `divide / get_pp_indices / IntermediateTensors / PPMissingLayer`         |
| `src/compute/layers/linear.py`         | `Column/Row/QKV/MergedColumnParallelLinear`（TP 分片）                       |
| `src/compute/layers/vocab_parallel.py` | `VocabParallelEmbedding` / `ParallelLMHead`                              |
| `src/compute/models/qwen3.py`          | TP 分片层 + PP `make_layers`/`PPMissingLayer`/`IntermediateTensors`         |
| `src/compute/worker.py`                | 单 rank 的 `init_environment / execute_model / destroy`                    |
| `src/compute/executor.py`              | `ModelExecutor`(单卡) / `MultiWorkerClient`(单机多卡) / `TorchrunExecutor`(多机) |
| `src/control/engine.py`                | 调度器只跑在主进程；通过 executor 驱动前向                                               |


- **单机多卡**：`MultiWorkerClient` 用 `multiprocessing` 每卡一 worker，`mp.Queue` 传
`ScheduledBatch`，只收 last-PP tp0 的采样结果。
- **多机**：`run_torchrun.py` 交给 `torchrun` 拉起，控制面（调度广播 + 采样回收）走
gloo，计算面（TP/PP 集合通信）走 NCCL。

---



## 正确性验证

```bash
# 单卡基线 / 单机 TP / PP（需对应数量空闲卡）
$PY test/test_parallel.py --tp 1 --pp 1
$PY test/test_parallel.py --tp 2 --pp 1
$PY test/test_parallel.py --tp 1 --pp 2
$PY test/test_parallel.py --tp 2 --pp 2   # 需 4 张卡
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