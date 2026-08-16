# v3 test

功能正确性测试，与性能 benchmark 分离。每个脚本独立可跑，任一断言失败返回非零退出码。

在 `vllm-v3/` 下：

```bash
# 全部
../.venv2/bin/python test/run_all.py

# 或逐个
../.venv2/bin/python test/test_sampler.py
../.venv2/bin/python test/test_engine.py
../.venv2/bin/python test/test_prefix.py
../.venv2/bin/python test/test_scheduler.py
```

| 脚本 | 覆盖模块 | 验证内容 |
|------|----------|----------|
| `test_sampler.py` | compute/sampler | greedy / 纯温度 / top_k / top_p / min_p / 混合 六分支正确性、固定 seed 可复现 |
| `test_engine.py` | control/engine、compute/model_runner、compute/cuda_graph | eager 可复现、decode CG 与 eager 等价、compile+CG 与 eager 等价、max_tokens 终止、prompt 截断、max_tokens 收敛、EOS smoke |
| `test_prefix.py` | data/prefix（hash/radix）、data/manager、data/block_manager | full hit 与 partial remap 后输出与无缓存一致 |
| `test_scheduler.py` | control/scheduler | mix 开/关时 dec@P 行为、长请求 chunk prefill 完成 |

公共件在 `common.py`：引擎工厂、等长 dummy、greedy 运行、warmup 与销毁。不含计时逻辑，计时归 `bench/`。
