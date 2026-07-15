# Multi-node / heterogeneous notes (v7)

## Global rank layout

```text
world_size = tp_size * pp_size
rank = pp_rank * tp_size + tp_rank
```

On multi-node, each node owns a contiguous slice of ranks **or** an arbitrary map
via `device_ids` + an external launcher. Teaching Engines still spawn all ranks
from one process when GPUs are local; for true multi-node:

1. Fix `MASTER_ADDR` / `MASTER_PORT` (`--nccl-host`, `--nccl-port`).
2. Launch one Engine/worker group per node with ranks remapped, or use torchrun
   and replace the spawn path (production).
3. Swap ZMQ `ipc://` for `tcp://` endpoints shared across nodes
   (`src/runtime/zmq_rpc.tcp_path`).

## Suggested validation on PRO6000 + A100 + 4×3090

```bash
# 1) Per-GPU smoke (pick physical id)
CUDA_VISIBLE_DEVICES=0 python run.py --model ~/huggingface/Qwen3-0.6B --eager --max-new-tokens 16

# 2) Homogeneous TP on 3090s only
CUDA_VISIBLE_DEVICES=2,3,4,5 python run.py --model ~/huggingface/Qwen3-0.6B \
  --tp-size 2 --device-ids 0,1 --rpc-backend zmq --eager

# 3) Heterogeneous PP: big GPU first stage (ViT/LLM bottom)
# Map A100=0, 3090=1 after CUDA_VISIBLE_DEVICES=
CUDA_VISIBLE_DEVICES=<a100>,<3090> python run.py --model ~/huggingface/Qwen3-0.6B \
  --pp-size 2 --device-ids 0,1 --rpc-backend zmq --eager

# 4) Large single-GPU on PRO6000 / A100 with bigger model
CUDA_VISIBLE_DEVICES=<pro6000> python run.py --model ~/huggingface/Qwen3-8B --eager

# 5) Radix hit rate: shared system prompt × N requests via OpenAI API
```

## PD (future)

Run two servers: `--pd-mode prefill` and `--pd-mode decode` with a real
`pd_kv_backend`. Until then collocated mode only.
