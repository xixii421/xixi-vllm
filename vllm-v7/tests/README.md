# Tests

```text
tests/
  helpers/       cfg / paths / parallel / moe_tiny
  unit/          CPU 快测（调度、execute、采样…）
  gpu/           单卡 e2e / CG / layers / VL / MoE
  distributed/   ≥2 GPU：text TP/PP、PP transport、MoE EP
```

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.

pytest tests/unit -q
pytest tests/gpu -q
pytest tests/distributed -q --tb=short
```

性能对比不在 tests 里，见：

```bash
PYTHONPATH=. python -m bench.run --suite e2e,ttft
PYTHONPATH=. python scripts/bench_vs_vllm_full.py --mode cg --suite text
```
