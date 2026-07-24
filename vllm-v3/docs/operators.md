# v3 算子优化（实现笔记）

对应瓶颈消除总表：[02-v3如何消除瓶颈](./02-v3如何消除瓶颈.md)。  
先读为何慢：[01-v2性能瓶颈](./01-v2性能瓶颈.md)。

v3 在 **同构的 CB 控制面** 上，把 ModelRunner 数据面换成高性能路径。

## 1. FlashInfer Attention（瓶颈 A）

`FlashInferBackend`（`src/layers/attention_backend.py`）：

- **Prefill**：`BatchPrefillWithPagedKVCacheWrapper` + `cu_seqlens_q/k`
- **Decode**：`BatchDecodeWithPagedKVCacheWrapper` + paged indptr/indices/last_page_len

`Attention`（`src/layers/attention.py`）：

1. Triton `store_kvcache` 写入 `[2, L, blocks, block_size, n_kv, head_dim]`
2. `wrapper.run(q, (k_cache, v_cache))` —— **无 gather**

`compute_paged_kv_metadata()` 从 `block_tables` + `seq_lens` 生成 FI 元数据。

## 2. Triton KV Write（瓶颈 C）

`store_kvcache`（`src/memory/kv_ops.py`）：按 `slot_mapping` 写扁平 slot。

```
slot = block_id * block_size + offset_in_block
```

## 3. 动态 KV 池

`ModelRunner._alloc_kv()`：按 `gpu_memory_utilization` sizing；`block_size` 须为 **256 的倍数**。

## 4. Decode CUDA Graph（瓶颈 B）

`CUDAGraphRunner`（`src/core/cuda_graph.py`）：

- static buffers；batch 阶梯 capture
- replay 前 `_replan_cg()` 更新 paged 元数据

启用：`enforce_eager=False` / `run.py --compile`。Prefill 仍 eager。

## 5. 输入与采样（瓶颈 E）

- `input_builder.py`：`pin_memory` + `non_blocking` H2D
- `sampler.py`：整 batch 采样

## 6. 层算子（瓶颈 D）

| 算子 | 路径 | 实现 |
| --- | --- | --- |
| RMSNorm | `layernorm.py` | Triton fused |
| SiluAndMul | `activation.py` | `torch.compile` |
| RoPE | `rotary.py` | Triton |
| Attention | `attention*.py` | FlashInfer |
| KV write | `kv_ops.py` | Triton |

## 7. 教学边界

已移植：FI / Triton / CG / 动态池 / pin H2D / batched sampler / hash+radix APC。  
未移植：TP/PP、多模态、生产级分层 KV 等。相对 v7 的下一步见 [optimization_guide.md](./optimization_guide.md)。
