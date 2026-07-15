# vllm-v2 Benchmark

## 入口

```bash
cd vllm-v2 && source ../vllm-v0/.venv/bin/activate

# 默认 / 推荐：多请求真实 Prefill（走 PagedKVCache 写读）
python -m bench.run_all --suite kv_vram --model Qwen3-1.7B --context-len 4096

# 自定义一组并发长度
python -m bench.run_all --suite kv_vram --model Qwen3-1.7B --context-len 4096 \
  --paged-lengths 128 256 512 128

# PagedAttention 相关：kv_vram + gather 扫长
python -m bench.run_all --suite paged --model Qwen3-1.7B --context-len 2048 \
  --lengths 128 512 1024 --warmup 3 --iters 10
```

## `kv_vram` 测的是什么

| 项目 | 说明 |
| --- | --- |
| **真实** | R 路请求在**共享分页池**上依次 Prefill，页表同时占用；每路再 Decode 1 步（gather） |
| **分页逻辑 MiB** | `占页数 × B × bytes_per_token`（本场景实际占页） |
| **连续对照 MiB** | `R × M × bytes_per_token`（若每请求按满上下文连续预留，**估算**） |
| **不是** | 旧版那种在引擎外单独 `torch.zeros` 假张量比大小 |

## 符号

- **R**：并发请求数  
- **T_i / 真实ΣT**：各请求真实长度及其和  
- **M**：`--context-len`  
- **B**：`--block-size`  
- **占页**：`Σ ceil(T_i / B)`

## 其它套件

`ttft` / `prefill` / `decode` / `e2e` / `vram` / `latency_seqlen` / `paged_overhead` —— 真实计时或峰值显存；`--suite all` 全跑。

## 输出

所有结果写入固定目录 ``bench/results/``（不再创建 ``{timestamp}_{model}/`` 子目录）：

| 文件 | 说明 |
| --- | --- |
| `meta.csv` | 最近一次运行元数据，含 `run_timestamp` |
| `report.md` | 汇总 Markdown；一次 `run_all` 多套件时追加章节 |
| `decode.csv` 等 | 各套件数据，运行该套件时覆盖 |
