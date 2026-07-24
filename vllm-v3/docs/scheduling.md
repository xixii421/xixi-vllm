# vllm-v3：调度与 Continuous Batching

**调度不是相对 v2 的性能主因。** v2 已把 CB / chunked / PD mix 做对；v3 控制面故意同构，慢点在算子 → [01-v2性能瓶颈](./01-v2性能瓶颈.md)。

## 三面（与 v2 同构）

```text
用户 → Engine
         → Scheduler(kv)          # 控制面：waiting/running、budget、mix
         → ModelRunner(model,kv)  # 数据面：组张量 + FlashInfer/CG + 采样
         → KVManager              # 资源层：页分配 + kv_cache 池 + 前缀
```

- `schedule()` → `(prefills, decodes)`
- `Engine.step`：`model_runner.run(prefills, decodes)`（先 decode 后 prefill，**两次** forward —— 瓶颈 G，教学保留）
- 默认 `mix_prefill_decode=True`（decode-first + 一块 prefill）

数据面怎么换掉 v2 瓶颈：[02-v3如何消除瓶颈](./02-v3如何消除瓶颈.md)；实现笔记：[operators.md](./operators.md)。
