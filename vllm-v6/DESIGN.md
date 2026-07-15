# vllm-v6 DESIGN

## Goal

Megatron-style TP + PP on top of FlashInfer continuous batching + Qwen3-VL.
Launch via `multiprocessing.spawn` + NCCL.

## Package layout

```text
src/
  config.py / sampling_params.py / context.py
  distributed/   process groups, collectives, PP types, math
  kernels/       Triton kv_ops
  layers/        Attention, linears, RoPE/MRoPE, norms
  scheduling/    Sequence, BlockManager, KV pool, Scheduler
  multimodal/    processor, positions, ViT tower, vision_dp
  models/        Qwen3 / Qwen3-VL + weight loader
  runtime/       Engine, ModelRunner, Executor, Worker, sampler
  cuda_graph/    decode + ViT CUDA Graph runners
  tools/         hub download / smoke
```

Dependency order (bottom → top):

`distributed` → `kernels`/`layers` → `scheduling` → `models`/`multimodal` → `runtime`/`cuda_graph`

## Parallelism

`global_rank = pp_rank * tp_size + tp_rank`

- TP: Column/Row/QKV parallel linears; vocab embedding/LM head
- PP: layer slice + IntermediateTensors send/recv
- VL encoder: `mm_encoder_tp_mode=data|weights` (data shards image batch)

## Multimodal MRoPE

- Apply: `layers/mrope.py`
- Build positions: `multimodal/positions.py`
- Worker→Engine sync: `runtime/mrope_sync.py`
