# vllm-v3：调度与 Continuous Batching

v3 **控制面与资源层对齐 v2**；差异在数据面换成 FlashInfer / Triton / 可选 CUDA Graph。

## 三面（与 v2 同构）

```text
用户 → Engine
         → Scheduler(kv)          # 控制面：waiting/running、budget、mix
         → ModelRunner(model,kv)  # 数据面：组张量 + FlashInfer/CG + 采样
         → KVManager              # 资源层：页分配 + kv_cache 池 + hash 前缀
```

- `schedule()` → `(prefills, decodes)`
- `Engine.step`：`model_runner.run(prefills, decodes)`（先 decode 后 prefill，两次 forward）
- 默认 `mix_prefill_decode=True`（decode-first + 一块 prefill）

## 相对 v2 数据面

| 项 | v2 | v3 |
| --- | --- | --- |
| Attention | gather + SDPA | FlashInfer |
| KV 写 | `index_copy_` | Triton scatter |
| Decode | eager | 可选 CUDA Graph |
| block_size | 16 | 256 |

算子细节见 [operators.md](./operators.md)。
