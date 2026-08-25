# vllm-v4 — TP / PP / DP

在 v3 单卡引擎上加并行。控制面做成 vLLM V1 的形状：`Engine → Client → EngineCore → Executor → Worker`。`dp=1,tp=1,pp=1` 也走同一套接口。

| | 切什么 | 通信 | 进程 |
| --- | --- | --- | --- |
| **TP** | 每层权重 / head / 词表 | NCCL all-reduce / gather | 同 Engine，每卡一进程 |
| **PP** | 连续若干层 | NCCL send/recv hidden+residual | 同上；`--pp-layer-counts` 可不均分 |
| **DP** | 请求 | ZMQ（本机 IPC） | 独立 Engine 副本，各有一份 Scheduler |

单 Engine 的 NCCL world = `tp × pp`。总 GPU ≈ `dp × tp × pp`。DP 不要和 `torchrun` 混用。

怎么选：模型单卡放得下、要吞吐 → `DP + --cg`。单卡放不下、同构卡 → `TP + --cg`。异构显存 → 不均分 PP。无 NVLink 的小模型不要做 TP。完整矩阵见 [docs/03-性能对比.md](docs/03-性能对比.md)。

环境安装见仓库根目录 [README.md](../README.md)。在 `vllm-v4/` 下执行下面的命令。

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

```bash
python test/test_parallel.py --tp 1 --pp 1
python test/test_parallel.py --tp 2 --pp 1
python test/test_parallel.py --tp 1 --pp 2
python test/test_parallel.py --model ~/huggingface/Qwen3-8B --tp 1 --pp 2
python test/test_parallel.py --model ~/huggingface/Qwen3-14B --pp 2 --pp-layer-counts 12,28
python test/test_dp.py --dp 2 --n-reqs 4
```

0.6B 的 TP=2 / PP=2 与 fp32 HuggingFace 逐 token 一致。8B 的 PP=2 与单卡 bit 级一致。对照请用 fp32：bf16 greedy 在 logits 打平时可能因求和顺序分叉。NCCL 不允许两 rank 共用一张卡。

吞吐基准：`python bench/bench_v4.py --tp 2 --cg`，说明见 [bench/README.md](bench/README.md)。

---

## 文档

| | |
| --- | --- |
| 三种并行怎么选 | [docs/00-并行环境.md](docs/00-并行环境.md) |
| 控制面与三条通道 | [docs/04-控制与并行.md](docs/04-控制与并行.md) |
| 两台机器实测 | [docs/03-性能对比.md](docs/03-性能对比.md) |
| 硬件 / 通信 | [docs/01-硬件互联.md](docs/01-硬件互联.md) · [docs/02-服务器通信测量.md](docs/02-服务器通信测量.md) |
| TP 线性层 / 词表 | [docs/06-linear.md](docs/06-linear.md) · [docs/05-词表.md](docs/05-词表.md) |

这版不做：OpenAI HTTP、MoE、视觉、量化、投机解码。那些在 v5 / v6。DP 只做本机副本，没有跨机数据并行。
