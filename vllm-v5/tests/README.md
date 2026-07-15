# vllm-v4 tests

Functional tests for each core module. Run from `vllm-v4/`:

```bash
source .venv/bin/activate   # or: .venv/bin/python -m pytest ...
python -m pytest tests/ -q
```

## Layout

| File | Covers |
|------|--------|
| `test_sampler.py` | greedy / mixed temp / top-k / top-p / min-p |
| `test_sequence.py` | Sequence lengths, slot helpers, SamplingParams |
| `test_block_manager.py` | alloc/free, prefix cache, append, hash collision |
| `test_scheduler.py` | prefill/decode, chunk, EOS, max_tokens=0, preempt, promote |
| `test_input_builder.py` | prefill/decode tensors & slot_mapping |
| `test_kv_ops.py` | Triton `store_kvcache` vs PyTorch |
| `test_layers.py` | SiluAndMul, RMSNorm, RoPE |
| `test_attention_backend.py` | paged KV metadata, separate FlashInfer workspaces |
| `test_engine_e2e.py` | eager/CG batch, clamp, dummy page, stream (needs `~/huggingface/Qwen3-0.6B`) |

GPU tests skip automatically when CUDA (or the model) is unavailable.
