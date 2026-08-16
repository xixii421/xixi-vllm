# v3 bench

性能测试，与功能正确性 test 分离。结果只写 bench/out/*.txt（纯文本，无 json/md）。

在 vllm-v3/ 下：

```bash
# 全套（推荐）：清空 out/ 后顺序跑；官方 vLLM 走 .venv-vllm
../.venv2/bin/python bench/run_all.py

# 或逐个
../.venv2/bin/python bench/bench_prefill.py
../.venv2/bin/python bench/bench_e2e.py
../.venv2/bin/python bench/bench_compare.py --engine v3
../.venv2/bin/python bench/bench_compare.py --engine nano
../.venv-vllm/bin/python bench/bench_compare.py --engine vllm
../.venv2/bin/python bench/bench_ttft_decode.py --engine v3
../.venv2/bin/python bench/bench_ttft_decode.py --engine nano
../.venv-vllm/bin/python bench/bench_ttft_decode.py --engine vllm
../.venv2/bin/python bench/bench_concurrent.py --engine v3
../.venv2/bin/python bench/bench_concurrent.py --engine nano
../.venv-vllm/bin/python bench/bench_concurrent.py --engine vllm
```

| 脚本 | 回答什么 | 覆盖模块 | 输出 |
|------|----------|----------|------|
| bench_prefill.py | 短/长 prompt 稳态 TTFT：eager vs compile | prefill / torch.compile | out/bench_prefill.txt |
| bench_e2e.py | eager / decode-CG / CG+compile 的连续批吞吐（隔离 CG 与 compile 边际收益） | decode CUDA Graph + compile | out/bench_e2e.txt |
| bench_compare.py | 同一 workload 对比 v3、nano-vllm、官方 vLLM 吞吐 | 端到端跨引擎 | out/bench_compare.txt |
| bench_ttft_decode.py | 规范 TTFT / TPOT；v3 radix vs nano/官方 hash 前缀命中 | TTFT/TPOT/前缀 | out/bench_ttft_decode.txt |
| bench_concurrent.py | 高并发 32-256（gpu_mem=0.90）吞吐对比 | 高并发连续批 | out/bench_concurrent.txt |

共享逻辑在 common.py：引擎工厂、等长 dummy、3 轮 greedy warmup（compile -> 录图 -> 重放），计时不含 tokenizer/print。输出统一为 txt：write_result 覆盖式写单个报告，append_lines 供跨引擎脚本各自追加一行、header 只写一次。

口径：

- TTFT = 第一个 completion token；greedy；warmup 之后取中位数
- Decode / 连续批吞吐见 bench_e2e.py、bench_compare.py
- bench_prefill 两边都开 decode CG，只隔离 torch_compile
- bench_ttft_decode 因 nano 禁止 greedy，三边统一 temperature=0.6、ignore_eos；TTFT / TPOT 对齐 vLLM bench serve（并发 1；prefix 场景 prefix=1024 / suffix=128）
- 三个引擎不能同进程混 import，跨引擎脚本按 --engine 分三次跑
