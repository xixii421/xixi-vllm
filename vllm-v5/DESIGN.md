# vllm-v7 Design

主题：在 v6（TP/PP + 多模态 + CUDA Graph）之上对齐 **ullm（μLLM）** 的控制面（ZMQ / Radix / Async API），并支持教学版 **Qwen3 MoE（cutlass + TP/PP）**。

## ZMQ vs Pipe

| | `multiprocessing.Pipe` | **ZMQ** |
|--|--|--|
| 形态 | OS 管道，一对一 | 消息套接字 |
| 拓扑 | 主进程 × N 个 Pipe | 每 rank PUSH→PULL + response |
| 跨机 | 不方便 | `tcp://` / `ipc://` |

**NCCL 只负责 GPU 张量**（TP all-reduce、PP send/recv）。  
ZMQ / Pipe 只传控制消息：`execute_model`、`shutdown`、KV block 数等。

```text
 API/Async ─► Engine/Scheduler ──ZMQ/Pipe──► Workers ──NCCL──► GPU kernels
```

`rpc_backend=zmq`（默认）或 `pipe`。

## 架构约定（教学要点）

- **依赖方向**（禁止反向）：`entrypoints → runtime → models → layers → distributed`；`config` / `sampling` / `utils` 为底层共享包
- **`runtime/` 按调用流水线分包**：
  ```text
  engine / async_llm     控制面
    → input/             Stage1 CPU 预处理
    → scheduling/        调度（外部包）
    → execute/           Stage2 RPC / Worker
    → runner/            Stage3 GPU forward → sample
         ↳ vision/       Stage3b VL（仅 prefill）
  ```
- 架构讲义：[`docs/README.md`](docs/README.md)

## TP / PP（MoE = cutlass 优先 + Triton 回退）

```text
rank = pp_rank * tp_size + tp_rank
world_size = tp_size * pp_size
```

- **TP**：同 PP stage 内宜同构（默认拒异构；仅教学可 `allow_hetero_tp=True`）。MoE 在 intermediate 维切分，forward 后 `all-reduce`。  
- **PP**：层级切分。  
- **MoE backend**：`moe_fused_backend=auto|cutlass|triton`  
  - **cutlass**（FlashInfer）：SM89/90/100+；SM120 需系统 nvcc≥12.9  
  - **triton**：精简 fused MoE，覆盖 A100/3090/消费级等 cutlass 不可用场景  
- 推荐真实权重：`Qwen1.5-MoE-A2.7B`（约 14B / 激活 2.7B）；测试亦可用 `tests/helpers/moe_tiny.py`。

Executor / Scheduler / BlockManager **不感知** MoE 内核选型。

## 目录

```text
src/           引擎核心
docs/          架构讲义（控制/计算、调度流、计算流）
bench/lib/     微基准核心库；入口 python -m bench.run
scripts/       精简 CLI：hub / bench_vs_vllm_full / save_sharded_state
tests/         unit / gpu / distributed（精简）
```

## CLI

```bash
PYTHONPATH=. python -m bench.run --suite e2e,ttft,decode
PYTHONPATH=. python scripts/bench_vs_vllm_full.py --mode cg --suite text,vl,moe
PYTHONPATH=. python scripts/hub.py --help
pytest tests/unit -q
```
