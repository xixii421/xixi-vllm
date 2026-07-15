# vllm-v7

主题：**ullm 风格控制面** — ZMQ RPC、Radix 前缀缓存、Async + OpenAI API、多机 NCCL 配置、PD 分离 stub。底座仍是 v6 的 TP/PP + Qwen3-VL + FlashInfer。

## 快速开始

```bash
cd vllm-v7
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.
source .venv/bin/activate   # → vllm-v6/.venv

# 单卡（Radix 默认开；ZMQ 仅 world_size>1 生效）
python run.py --model ~/huggingface/Qwen3-0.6B --temperature 0 --max-new-tokens 64 --eager

# TP=2 + ZMQ（默认）
python run.py --model ~/huggingface/Qwen3-0.6B \
  --tp-size 2 --device-ids 0,1 --rpc-backend zmq --eager

# 回退到 v6 Pipe
python run.py --model ~/huggingface/Qwen3-0.6B \
  --tp-size 2 --device-ids 0,1 --rpc-backend pipe --eager

# OpenAI API
python -m src.entrypoints.openai_api --model ~/huggingface/Qwen3-0.6B --port 8000 --eager
```

## 架构（相对 v6）

```text
OpenAI / AsyncLLM
       │
   Engine + Scheduler + RadixBlockManager
       │  ZMQ PUB  (或 Pipe)
   Workers (TP×PP) ── NCCL tcp://nccl_host:port
       │
   (optional PD stub) Prefill ↔ Decode KV transfer
```

详见 [DESIGN.md](DESIGN.md)。异构 / 多机验证步骤也在其中。

## 测试

```bash
./.venv/bin/python -m pytest -m unit -q
./.venv/bin/python -m pytest tests/unit/test_radix.py -q
# 双卡
./.venv/bin/python -m pytest -m distributed -q --tb=short
```

## 依赖增量

`pyzmq`、`fastapi`、`uvicorn`、`pydantic`（已写入 `requirements.txt`）。
