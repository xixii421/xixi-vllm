# vllm-v4 — TP / PP / DP

支持单卡推理与张量 / 流水线 / 数据并行。控制面做成 vLLM V1 的形状：`Engine → Client → EngineCore → Executor → Worker`。`dp=1,tp=1,pp=1` 也走同一套接口。

| | 切什么 | 通信 | 进程 |
| --- | --- | --- | --- |
| **TP** | 每层权重 / head / 词表 | NCCL all-reduce / gather | 同 Engine，每卡一进程 |
| **PP** | 连续若干层 | NCCL send/recv hidden+residual | 同上；`--pp-layer-counts` 可不均分 |
| **DP** | 请求 | ZMQ（本机 IPC） | 独立 Engine 副本，各有一份 Scheduler |

单 Engine 的 NCCL world = `tp × pp`。总 GPU ≈ `dp × tp × pp`。DP 不要和 `torchrun` 混用。

怎么选：模型单卡放得下、要吞吐 → `DP + --cg`。单卡放不下、同构卡 → `TP + --cg`。异构显存 → 不均分 PP。无 NVLink 的小模型不要做 TP。完整矩阵见 [docs/03-性能对比.md](03-性能对比.md)。

环境安装见仓库根目录 [README.md](../README.md)。在仓库根目录执行下面的命令；先运行 `source .venv/bin/activate` 激活根目录环境，也可为各条命令添加 `uv run` 前缀。

---

## 代码架构

```
./
├── run.py                        # --tp / --pp / --dp / --pp-layer-counts / --device-ids
├── run_torchrun.py               # 多机 torchrun 入口
├── config.yaml                   # tp_size / pp_size / dp_size / nccl 等
├── src/
│   ├── config.py                 # tp/pp/dp/pp_layer_counts/device_ids/nccl_port
│   ├── control/                  # 多进程控制面
│   │   ├── client.py             #   Client 屏蔽调用差异
│   │   ├── core.py               #   EngineCore
│   │   ├── executor.py           #   Executor
│   │   └── mp_client.py          #   多进程 Client
│   ├── data/
│   │   └── kv.py                 # driver 侧 KV 账本
│   ├── compute/                  # worker / kv_pool / 并行层
│   │   ├── batch.py              #   GPU 侧 batch
│   │   ├── kv_ops.py             #   KV Cache 写入算子
│   │   ├── kv_pool.py            #   GPU 上的 KV 张量池
│   │   ├── model_runner.py       #   按 rank 组织前向
│   │   ├── worker.py             #   每卡 Worker
│   │   ├── layers/
│   │   │   ├── linear.py         #   列并行 / 行并行 Linear
│   │   │   ├── vocab_parallel.py #   词表并行
│   │   │   └── attention.py      #   TP 切 head
│   │   └── models/
│   │       └── qwen3.py          #   接入并行层
│   └── distributed/              # 并行通信
│       ├── parallel_state.py     #   rank / 进程组管理
│       ├── communication_op.py   #   all-reduce / all-gather 封装
│       ├── control_channel.py    #   控制通道
│       └── utils.py              #   通信辅助
├── test/                         # 并行正确性测试
├── bench/                        # 吞吐基准
└── docs/                         # 设计文档
```

---

## 核心能力

| 能力 | 对应模块 | 说明 |
|---------|---------|------|
| 多进程控制面 | `src/control/client.py` / `core.py` / `executor.py` / `mp_client.py` | Engine → Core → Executor → Worker 分层，单卡多卡同接口 |
| 张量并行 TP | `src/compute/layers/linear.py` / `vocab_parallel.py` | 列并行 / 行并行线性层 + 词表并行，all-reduce / all-gather |
| 流水线并行 PP | `src/control/core.py` / `executor.py` / `compute/worker.py` | 按层切 stage，send/recv hidden，支持不均分层数 |
| 数据并行 DP | `src/control/client.py` / `mp_client.py` | 独立 Engine 副本 + round-robin 分发，ZMQ 本机 IPC |
| 通信层 | `src/distributed/` | NCCL（TP/PP）、ZMQ（DP）、Gloo（控制） |
| GPU KV 池拆分 | `src/compute/kv_pool.py` | 真正 KV 张量放到 compute 侧，账本留在 data 侧 |

---

## 启动

默认是 eager，启动快。要吞吐加 `--cg`（decode CUDA Graph）；prefill 再加 `--compile`。

```bash
# 单卡
python run.py --model Qwen3-0.6B
python run.py --model Qwen3-0.6B --cg --compile

# 机内 TP / PP（每卡一进程，主进程 spawn）
python run.py --model Qwen3-0.6B --tp 2 --cg
python run.py --model Qwen3-8B --pp 2 --cg
python run.py --model Qwen3-14B --pp 2 --pp-layer-counts 12,28 --device-ids 1,0

# DP（独立副本，round-robin 分请求）
python run.py --model Qwen3-0.6B --dp 2 --cg --batch
python run.py --model Qwen3-8B --dp 2 --tp 2 --device-ids 0,1,2,3 --cg --batch

# 批量演示 / 指定 GPU
python run.py --model Qwen3-0.6B --batch --cg
CUDA_VISIBLE_DEVICES=1,2 python run.py --model Qwen3-0.6B --tp 2 --cg
```

常用参数：`--tp` `--pp` `--dp` `--pp-layer-counts` `--device-ids` `--cg` `--compile` `--prefix-backend none|hash|radix`。其余在 `config.yaml`。`--device-ids` 长度必须等于 `dp×tp×pp`。

## 测试

测试通过独立脚本运行，不使用 pytest 收集。

```bash
python test/test_sharding.py
python test/test_pp_comm.py
python test/test_parallel.py --tp 1 --pp 1
python test/test_parallel.py --tp 2 --pp 1
python test/test_parallel.py --tp 1 --pp 2
python test/test_parallel.py --model ~/huggingface/Qwen3-8B --tp 1 --pp 2
python test/test_parallel.py --model ~/huggingface/Qwen3-14B --pp 2 --pp-layer-counts 12,28
python test/test_dp.py --dp 2 --n-reqs 4
```

0.6B 的 TP=2 / PP=2 与 fp32 HuggingFace 逐 token 一致。8B 的 PP=2 与单卡 bit 级一致。对照请用 fp32：bf16 greedy 在 logits 打平时可能因求和顺序分叉。NCCL 不允许两 rank 共用一张卡。

吞吐基准：`python bench/bench_v4.py --tp 2 --cg`，说明见 [bench/README.md](../bench/README.md)。

---

## 文档

| 文档 | 说明 |
|------|------|
| [00-并行环境.md](00-并行环境.md) | 三类并行策略与选型 |
| [01-硬件互联.md](01-硬件互联.md) | 单机 GPU 拓扑与互联 |
| [02-服务器通信测量.md](02-服务器通信测量.md) | 四卡 A100 卡间通信实测 |
| [03-性能对比.md](03-性能对比.md) | 两台机器上 TP / PP / DP 与 CG / compile |
| [04-控制与并行.md](04-控制与并行.md) | 控制面与 Gloo / NCCL / ZMQ |
| [05-词表.md](05-词表.md) | 词表并行原理与实现 |
| [06-linear.md](06-linear.md) | 列并行 / 行并行线性层 |

当前支持 Qwen3 Dense 推理与 TP / PP / DP；DP 仅支持本机副本。
