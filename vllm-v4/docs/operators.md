# v4 算子优化

v4 在 v3 调度层不变的前提下，把 **ModelRunner 数据面**换成与 `dzyy_vllm` 对齐的高性能路径。

## 1. FlashInfer Attention

`FlashInferBackend`（`src/layers/attention_backend.py`）管理两个 wrapper：

- **Prefill**：`BatchPrefillWithPagedKVCacheWrapper`，配合 `cu_seqlens_q/k` 做 varlen batch
- **Decode**：`BatchDecodeWithPagedKVCacheWrapper`，配合 paged KV indptr/indices/last_page_len

`Attention` 层（`src/layers/attention.py`）流程：

1. Triton `store_kvcache` 把本步 K/V scatter 到 `[2, L, blocks, block_size, n_kv, head_dim]` 池
2. `wrapper.run(q, (k_cache, v_cache))` 执行 fused attention

`compute_paged_kv_metadata()` 从 `block_tables` + `seq_lens` 生成 FlashInfer 所需元数据。

## 2. Triton KV Write

`store_kvcache`（`src/memory/kv_ops.py`）每个 token 一个 program，按 `slot_mapping` 写入扁平化 slot：

```
slot = block_id * block_size + offset_in_block
```

比 PyTorch `index_copy_` 更适合大批量 scatter，且与 FlashInfer NHD layout 对齐。

## 3. 动态 KV 池

`ModelRunner._alloc_kv()`：

- 读取 `torch.cuda.mem_get_info`
- 按 `gpu_memory_utilization`（默认 0.9）计算可用显存
- 分配 `[2, num_layers, num_blocks, block_size, n_kv_heads, head_dim]`
- `EngineConfig.num_kvcache_blocks > 0` 时可固定块数（bench 用）

`block_size` 必须为 **256 的倍数**（FlashInfer / flash-attn 约定）。

## 4. Decode CUDA Graph

`CUDAGraphRunner`（`src/core/cuda_graph.py`）：

- 预分配 static buffers（input_ids, positions, slot_mapping, block_tables, …）
- 按 batch size 阶梯 `[1,2,4,…,256]` capture graph
- replay 前 `_replan_cg()` 更新 paged KV metadata（indptr/indices）

启用：`EngineConfig(enforce_eager=False)` 或 `python run.py --compile`。

Prefill 仍走 eager FlashInfer（shape 变化大，不适合 CG）。

## 5. 输入管线

`input_builder.py` 使用 **pin_memory + non_blocking** H2D：

- Prefill：拼接 varlen `input_ids`/`positions`，`cu_seqlens_q/k`，`slot_mapping`
- Decode：每 seq 1 token，`context_lens`，`block_tables`

`model_runner._logits_for_sample()` 在 prefill 用 `cu_seqlens_q[1:]-1` 取每 seq 最后 token hidden。

## 6. 批量采样

`Sampler`（`src/engine/sampler.py`）：

- `temperature ≈ 0` → `argmax`
- 否则 Gumbel-max style sampling（与 dzyy_vllm 一致）

## 7. 层算子加速（与 dzyy_vllm 对齐）

| 算子 | 路径 | 实现 |
| --- | --- | --- |
| RMSNorm | `layers/layernorm.py` | Triton fused（含 residual add） |
| SiluAndMul | `layers/activation.py` | `torch.compile` |
| RoPE | `layers/rotary.py` | Triton in-place（CUDA）；CPU fallback |
| Attention | `layers/attention*.py` | FlashInfer |
| KV write | `memory/kv_ops.py` | Triton scatter |

## 8. 与 dzyy_vllm 的差异

v4 **未移植**（教学版刻意精简）：

- Tensor Parallel / Pipeline Parallel
- 多模态 / MRoPE
- Radix cache / 更复杂 preempt 策略

v4 **已移植**：

- FlashInfer backend
- Triton KV ops / RMSNorm / RoPE
- torch.compile SiluAndMul
- CUDA Graph decode
- 动态 KV pool
- pin_memory input pipeline
- batched sampler
- `block_size=256`
