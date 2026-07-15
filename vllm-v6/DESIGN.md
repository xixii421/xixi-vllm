# vllm-v6 DESIGN

## Goal

Teach **Megatron-style Tensor Parallelism (TP) + Pipeline Parallelism (PP)** on top of
v5 (FlashInfer + continuous batching + Qwen3-VL). Launch via `multiprocessing.spawn`
+ NCCL (not `torchrun`).

## Parallelism model

```text
global_rank = pp_rank * tp_size + tp_rank
WORLD  — all ranks
TP     — ranks that shard the same layer (same pp_rank)
PP     — ranks that own consecutive layer ranges (same tp_rank)
```

- **TP**: Column/Row/QKV/Merged parallel linears; VocabParallelEmbedding / ParallelLMHead;
  Attention uses **local** `num_heads = total // tp_size` and local KV heads.
- **PP**: `make_layers` + `PPMissingLayer` + `get_pp_indices`; mid-stages exchange
  `IntermediateTensors{hidden_states, residual}` via PP send/recv (with TP all-gather
  of activation slices).
- **Device map**: Worker rank `r` uses `device_ids[r]` so heterogeneous setups
  (e.g. 4070+4090) work without remapping `CUDA_VISIBLE_DEVICES`.

## Components

| Path | Role |
| --- | --- |
| `src/distributed/` | `parallel_state`, `communication_op`, `get_pp_indices` |
| `src/layers/linear.py` | Column / Row / QKV / Merged parallel linears |
| `src/layers/vocab_parallel_embedding.py` | Vocab + LM head sharding |
| `src/layers/utils_parallel.py` | `IntermediateTensors`, `PPMissingLayer`, `divide` |
| `src/engine/worker.py` | Per-rank NCCL init, model, PP recv/send |
| `src/engine/executor.py` | Spawn `tp×pp` workers; Pipe RPC; driver = last PP & tp0 |
| `src/engine/engine.py` | `world_size==1` → ModelRunner; else → Executor |

## Single-GPU compatibility

When `tp_size=1, pp_size=1`, Engine keeps the in-process ModelRunner path but still
calls `initialize_model_parallel(1,1)` so parallel layers behave as replicated
(linears are full-size; collective ops are no-ops).

## Multimodal

Qwen3-VL supports **TP and PP**. Vision tower is replicated on every first-PP TP
rank; language weights are sharded. Mid/last PP ranks skip the ViT to save VRAM.

## CUDA Graph

Decode CUDA Graph captures **local stage compute** (`pp_role`: full / first /
middle / last). PP send/recv stay outside the graph (Worker path). Prefer
`--compile` (or `enforce_eager=False`) for both text and VL.
## Heterogeneous GPUs

Rank `r` uses `device_ids[r]`. TP stays closer to bit-exact because every layer
all-reduces across devices. PP can drift after a few decode steps when stage-0
and stage-1 run on different GPU models (e.g. 4070 vs 4090) — prefill boundary
activations still match; late greedies may not. E2E tests check a short prefix for PP.
