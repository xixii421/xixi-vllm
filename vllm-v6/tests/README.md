# vllm-v6 tests

```text
tests/
  helpers/       shared Engine helpers
  unit/          CPU fast
  gpu/           1×CUDA
  distributed/   ≥2 GPU NCCL
  scripts/       manual HF compare (not collected)
```

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.
pytest -m unit -q
pytest -m "unit or gpu" -q
pytest -m distributed -q --tb=short
```
