# dzyy-vllm 教学推理框架

本仓库以「从零手写 vLLM」为主线，用多个可独立运行、可测量、有完整博客和注释的版本，逐步覆盖工业级推理系统的核心能力。教程版代码位于 `vllm-v0/` 至 `vllm-vX/`，每一版都有可复现的 profile 报告。


## 仓库定位与演进逻辑

vLLM 类推理框架要解决的本质矛盾，是**正确性、吞吐、显存效率、系统复杂度**四者之间的权衡。原生 HuggingFace 循环能跑通模型，但在长上下文、多请求并发、大模型分布式部署场景下，会同时遭遇计算冗余、显存碎片和静态批处理三大瓶颈。本仓库不试图在一个目录里堆叠全部功能，而是**每个版本只攻克一个瓶颈**，并以上一版的测试与 profile 数据作为下一版的起点。

`vllm-v0` 证明模型骨架与权重加载是正确的。`nano-vllm` 约 1300 行代码，是「教学版推理引擎」的达标参照，其核心标志是 Scheduler、BlockManager、ModelRunner 三层分离，以及 PagedAttention 与 Continuous Batching 同时成立。`dzyy_vllm` 约 5000 行，在 nano-vllm 能力之上叠加 FlashInfer、CUDA Graph、Radix 前缀缓存、多模态、TP/PP 与投机解码，对应本教程 v4 至 v7 的进阶目标。`vllm` 上游则涵盖全模型生态与生产调度，超出本教程范围，仅作最终参照。

合理的写法是**每版一个目录、一份 DESIGN、一套 bench/profile、一篇 docs**。不要在单目录内用开关堆功能，否则博客叙事与性能归因都会失真。接口层面可以提前预留（例如 Attention 接受 `kv_cache=None`、Engine 外置 `positions`），但实现必须按版本增量交付。

---

## 七版迭代路线

下表给出各版本的合理边界。每版「实现什么」与「不实现什么」同等重要，边界清晰才能保持教学可读性。

| 版本 | 核心主题 | 应实现的能力 | 本版不做 | 达标参照 |
| :--- | :--- | :--- | :--- | :--- |
| **v0** | 朴素自回归 | Qwen3 模型、权重加载、Sampler、单请求流式 `generate`、多尺寸权重验证 | KV Cache、批处理、调度、分布式 | 输出与 HF greedy 一致 |
| **v1** | KV Cache | 单请求 Prefill/Decode 分离，Attention 读写 cache，每步只算 1 个新 token | 分页、多请求、调度器 | Decode tok/s 比 v0 高 5× 以上 |
| **v2** | PagedAttention | Block 池、block table、非连续 KV 存储 | Continuous Batching | 多请求总显存低于静态预分配 50% |
| **v3** | 调度与并发 | Scheduler、ModelRunner、LLMEngine 拆分，FCFS，Continuous Batching | CUDA Graph、分布式 | 对齐 nano-vllm 架构，并发吞吐显著提升 |
| **v4** | 算子与图优化 | FlashInfer 后端，Decode 阶段 CUDA Graph，`enforce_eager` 可切换 | 多模态、TP/PP | 单请求 Decode 比 v3 再快 3× 以上 |
| **v5** | 多模态 | Qwen3-VL 最小闭环，ViT 编码，MRoPE，图文联合 Prefill | 分布式 | 图文问答结果正确 |
| **v6** | 分布式 | TP（优先），PP（单层流水），多卡加载 8B+ 模型 | PD 分离 | 多卡输出与单卡 greedy 一致 |
| **v7** | 系统级加速 | Radix 前缀缓存，n-gram 投机解码，PD 分离配置与接口预留 | 全量多机生产部署 | 共享 prompt 场景 Prefill 算力节省 80% 以上 |

### v0 至 v3（走向 nano-vllm）

v0 的价值在于**证明模型实现无误**，而非追求速度。Engine 每步将完整序列送入 Transformer，Attention 复杂度随序列长度二次增长，这是故意保留的「反面教材」，为 v1 的 KV Cache 提供可量化的优化空间。v0 应已完成 `model` 与 `lm_head` 分离、`positions` 外置传入、QKV 合并加载、weight tying 等接口预留。

v1 只解决**单请求重复计算**问题。引入 `Sequence` 与每层 KV 张量，Prefill 阶段一次性写入，Decode 阶段仅传入最后一个 token 及其位置。此时尚可不引入 block 分页，KV 仍可用连续张量，以降低学习成本。

v2 解决**显存碎片**问题。将 KV 池改为 `[2, L, N_blocks, block_size, H_kv, D]` 形态，每请求持有 block table。v2 可用 PyTorch 索引实现 PagedAttention，性能不是重点，正确性优先。

v3 解决**多请求调度**问题，也是本教程的第一次架构重构。`Engine.generate` 中的 for 循环应上移至 `LLMEngine.step()`，由 Scheduler 维护 waiting/running 队列，ModelRunner 负责组 batch 与调用模型。v3 完成即达到 nano-vllm 核心水准，应与 `nano-vllm/bench.py` 做同配置对比。

### v4 至 v7（对齐 dzyy_vllm 进阶能力）

v4 将瓶颈从算法转向**系统开销**。Python 逐步调度、kernel launch、非融合 Attention 会成为主因，需引入 FlashInfer 与 Decode 阶段 CUDA Graph。Prefill 与 Decode 应使用不同 kernel 路径，与 dzyy_vllm 的 `attention_backend.py` 和 `core/cuda_graph.py` 思路一致，但教程版保持更小表面。

v5 引入**多模态维度**，建议独立分支开发，避免与 v4 性能代码纠缠。最小路径为图片经 ViT 得 visual tokens，插入 text embedding 序列，经 MRoPE 后走 v3 已有调度与 KV 路径。

v6 处理**模型放不进单卡**的问题。先实现 TP（列并行 + 行并行 + all-reduce），再实现 PP（按层切分 + send/recv）。教程版不必追求极致 overlap，但需保证数值一致。

v7 处理**重复计算与延迟的最后一块**。RadixTree 前缀缓存让相同 system prompt 共享 KV block；n-gram 投机解码在 greedy 下无损加速；PD 分离本版只做配置项与 KV 传输接口 stub，不要求真多机。

---

## 每版必须有的 Profile 体系

教学的核心不仅是「能跑」，更是「能用数据讲清楚为什么需要下一版」。因此每个版本目录下应固定包含 `bench/`（宏观指标）与 `profile/`（微观归因），并在 `docs/` 中用同一次运行的数据写博客。

### 宏观指标（每版必做）

宏观 profile 回答「端到端快不快、费不费显存」。业界推理框架与 `dzyy_vllm/bench/bench_full.py` 普遍采用以下维度，教程版从 v0 起就应全部覆盖，即使 v0 的绝对数值较慢。

| 指标 | 含义 | 测量要点 |
| :--- | :--- | :--- |
| **TTFT** | 首 token 延迟 | 从请求进入到第一个 token 输出的时间，含 Prefill |
| **Prefill 延迟** | 提示词处理时间 | 固定 `max_tokens=1`，扫不同 prompt 长度 |
| **Decode 吞吐** | 纯生成速度 | `tok/s`，仅用生成阶段耗时除以 token 数 |
| **E2E 吞吐** | 端到端速度 | 含 Prefill 的总 `tok/s`，便于和 HF 对比 |
| **Peak VRAM** | 峰值显存 | `torch.cuda.max_memory_allocated()` |
| **Latency vs SeqLen** | 延迟随长度变化 | v0 应呈上升趋势，v1 后 Decode 应趋于平坦 |

测量时需调用 `torch.cuda.synchronize()` 后再读时钟，否则 GPU 异步执行会导致数据虚假偏低。使用 `time.perf_counter()` 即可，无需复杂基础设施。每个版本将结果写入 `bench/results_v{N}.json` 或 Markdown 表格，并保留硬件型号、模型名、batch size、dtype 等元数据。

### 微观归因（按版本递进）

微观 profile 回答「时间花在哪个算子、哪段 Python 逻辑」。主流方案分为四层，教程应逐层掌握。

**第一层是 PyTorch Profiler（`torch.profiler`）**。这是入门首选，可在 Chrome trace（`chrome://tracing`）或 Perfetto UI 中查看 CPU 与 CUDA 时间线，看到 `aten::matmul`、`scaled_dot_product_attention` 等算子占比。v0 应用它对比 `seq_len=128` 与 `seq_len=512` 单步 forward，直观展示 Attention 随长度增长的开销，为 v1 提供论据。开启方式建议 `activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]`，`record_shapes=True`，对单步或整段 `generate` 包一层 `with torch.profiler.profile(...)`。

**第二层是 NVTX 标注（`torch.cuda.nvtx`）**。在 `Engine.generate` 的 Prefill 等价段、单步 forward、采样、`tokenizer.decode` 等处插入 `nvtx.range_push/pop`，可在 PyTorch Profiler 或 Nsight Systems 时间线上看到具名区段。v0 应至少标注 `forward`、`compute_logits`、`sample`、`decode` 四段，用于证明 v0 瓶颈在 GPU forward 而非 Python 采样。

**第三层是 NVIDIA Nsight Systems（`nsys`）**。系统级时间线，适合观察 CPU 调度、CUDA kernel 发射、拷贝是否与计算重叠。命令形如 `nsys profile python run.py`。v0 用它检查每步是否因 `torch.tensor([input_ids])` 和 `.item()` 产生过多 CPU 同步。v4 之后应用它验证 CUDA Graph 是否减少 launch 空洞。

**第四层是 Nsight Compute（`ncu`）**。单 kernel 深度分析，寄存器、带宽、occupancy。教程中 v4 引入自定义 Triton 或 FlashInfer kernel 后再使用，v0 不必强求。

此外还有 **`torch.cuda.memory._record_memory_history`** 与 **`cudaMalloc` 快照** 用于查显存泄漏，v2 引入 BlockManager 后应每版运行一次内存回归。

### 各版本 Profile 重点

| 版本 | 宏观重点 | 微观重点 | 与上一版对比 |
| :--- | :--- | :--- | :--- |
| v0 | TTFT、Decode tok/s、Latency vs SeqLen（应上升） | Attention 占比、每步 `torch.tensor` 与 `.item()` 开销 | 基线 |
| v1 | Decode tok/s 大幅提升，Latency vs SeqLen Decode 段变平 | Prefill 与 Decode 分阶段 NVTX | 证明 KV Cache 消灭重复计算 |
| v2 | 多请求 Peak VRAM、显存利用率 | Block 分配次数、碎片率 | 证明分页省显存 |
| v3 | 并发吞吐 vs batch size、短请求不被长请求阻塞 | Scheduler step 耗时、batch 组装 | 对标 nano-vllm |
| v4 | Decode tok/s 跃升 | CUDA Graph capture 成功率、kernel 占比 | 对标 dzyy_vllm CG 收益 |
| v5 | 图文 TTFT、ViT 耗时占比 | ViT forward vs LLM forward | 多模态瓶颈定位 |
| v6 | 多卡线性度、通信占比 | all-reduce / P2P 耗时 | TP 效率 |
| v7 | 共享 prompt Prefill 加速比、投机加速比 | Radix 命中、accept rate | 系统优化收益 |

---

## v0 版本应如何 Profile

v0 是当前**朴素循环**基线；**v1 已实现 KV Cache**，可在同 prompt 下与 v0 对比 Decode 吞吐与 Latency 曲线。

第一类现象是 **Decode 延迟随序列长度上升**。因为每步对全长序列做 Attention，单步耗时应近似随 `seq_len` 线性至二次增长。建议在 `bench/profile_v0.py` 中对 `seq_len ∈ {64, 128, 256, 512, 1024}` 各测一步 greedy forward 耗时（warmup 后取中位数），绘制「单步延迟 vs 序列长度」表。该表是 v0 博客的核心图表。

第二类现象是 **Attention 占 forward 绝大部分时间**。用 PyTorch Profiler 抓取 `seq_len=256` 时一步 forward 的 CUDA 时间，导出 `key_averages().table(sort_by="cuda_time_total")`。预期 `scaled_dot_product_attention` 或与 `qkv_proj` 相关的 matmul 位居前列。若 Python 侧 `torch.tensor` 或 `arange` 占比异常高，说明 CPU 构造输入也是优化点，但 v0 主矛盾仍在 GPU Attention。

第三类现象是 **显存随加载模型和序列长度增长，但无 KV 池化管理**。记录 `load` 后显存（权重）、`seq_len=512` 单步后 `max_memory_allocated`（权重 + 激活）。v0 没有 KV Cache，显存不会随生成步数持续线性暴涨，但每步 activation 与全长成正比。这与 v1 引入 cache 后的显存曲线形成对比。

v0 推荐的固定 benchmark 配置如下表，写入 `vllm-v0/bench/` 并在每次改代码后回归。

| 配置项 | 建议值 |
| :--- | :--- |
| 模型 | `Qwen3-0.6B`（主），`Qwen3-1.7B` / `Qwen3-4B` 做正确性验证 |
| 硬件记录 | GPU 型号、驱动、CUDA、PyTorch 版本 |
| Prompt | `"Hello, my name is"` 与 512 token 长 prompt 各一组 |
| 生成长度 | 128 / 256 tokens |
| 采样 | greedy（`temperature=0`）保证可复现 |
| 模式 | `enforce_eager=True`（默认），可选对比 `torch.compile` |
| 指标 | TTFT、Decode tok/s、E2E tok/s、Peak VRAM、单步延迟曲线 |

v0 微观 profile 的推荐执行顺序如下。先用 `run.py` 或 bench 脚本产出宏观表并保存 JSON。再对单步 forward 运行 PyTorch Profiler，导出 `trace_v0_seq256.json` 供 Perfetto 打开。随后在 `engine.py` 的 `generate` 循环内加入 NVTX 区段，用 `nsys profile` 跑 32 步生成，查看 `sample` 与 `forward` 的时间比例。若已安装 Nsight Systems，一份 nsys 报告即可支撑「v0 瓶颈在全长 Attention」的结论。

v0 不必引入 FlashInfer、CUDA Graph 或分布式 profiler。那些工具在 v4、v6 才有意义。v0 profile 的价值在于**建立统一的测量规范**，使 v1 优化后的同一张表可以直接叠放对比，读者一眼看出 KV Cache 带来的 Decode 曲线拉平。

---

## 目录结构约定

```
dzyy-vllm/
├── README.md                 本文件，总路线与 profile 规范
├── vllm-v0/                  当前版本，朴素循环
│   ├── src/
│   ├── docs/
│   ├── bench/                宏观指标脚本与结果
│   └── profile/              Profiler trace、nsys 报告说明
├── vllm-v1/ … vllm-v7/       按版递增，各自独立
├── dzyy_vllm/                完整参考实现与对标 bench
├── nano-vllm/                nano-vllm 基线
└── vllm/                     上游 vLLM
```

每个 `vllm-vN/` 在发布时应满足三件套，**可运行的 `src/`**、**可复现的 `bench/` + `profile/`**、**可阅读的 `docs/`**。博客文章引用本版 profile 表格中的具体数字，而非主观描述「快了很多」。

---

## 快速开始（v0 / v1）

**v0**（朴素循环，用于对比基线）：

```bash
cd vllm-v0 && source .venv/bin/activate
python run.py --model Qwen3-0.6B --temperature 0
python -m bench.run_all --suite decode
```

**v1**（KV Cache，Decode 显著加速）：

```bash
cd vllm-v1 && source ../vllm-v0/.venv/bin/activate
python run.py --model Qwen3-0.6B --temperature 0
python -m bench.run_all --suite decode
```

**v0 vs v1 对比**（在仓库根目录）：

```bash
python scripts/compare_v0_v1_bench.py --quick
```

设计说明见 `vllm-v1/docs/kv_cache.md`。

**v2**（PagedAttention：Block 池 + block_table + gather 教学实现）：

```bash
cd vllm-v2 && source ../vllm-v0/.venv/bin/activate
python run.py --model Qwen3-0.6B --prompt "Hello" --max-new-tokens 64 --temperature 0
python test/test_paged_vs_v1.py --model Qwen3-0.6B --max-new-tokens 32
python -m bench.run_all --suite kv_vram --model Qwen3-1.7B --context-len 4096
```

设计说明见 `vllm-v2/docs/paged_attention.md`（不做 Continuous Batching，留给 v3）。

**v3**（Scheduler + Continuous Batching + 块哈希前缀）：

```bash
cd vllm-v3 && source ../vllm-v0/.venv/bin/activate
python run.py --model Qwen3-0.6B --prompt "Hello" --max-new-tokens 64 --temperature 0
python run.py --model Qwen3-0.6B --batch --max-new-tokens 32
python test/test_scheduler.py
```

设计说明见 `vllm-v3/docs/scheduling.md`。

**v4**（FlashInfer + CUDA Graph + Triton KV + 动态 KV 池）：

```bash
cd vllm-v4 && bash setup_env.sh && source .venv/bin/activate
python run.py --model Qwen3-0.6B --prompt "Hello" --max-new-tokens 64 --temperature 0
python run.py --model Qwen3-0.6B --compile --max-new-tokens 64   # Decode CUDA Graph
python run.py --model Qwen3-0.6B --batch --max-new-tokens 32
```

设计说明见 `vllm-v4/docs/operators.md`（调度层同 v3，见 `vllm-v3/docs/scheduling.md`）。

**v3 vs v4 性能对比**（TTFT / decode / e2e / batch）：

```bash
# 需 flashinfer 环境，例如 dzyy_vllm/.venv
source dzyy_vllm/.venv/bin/activate
python scripts/compare_v3_v4_bench.py --quick
python scripts/compare_v3_v4_bench.py --model Qwen3-0.6B --with-cuda-graph
```

**v4 vs nano-vllm**（同套件；环境需同时有 `flashinfer` + `flash-attn`）：

```bash
source dzyy_vllm/.venv/bin/activate
python scripts/compare_v4_nano_bench.py --quick
python scripts/compare_v4_nano_bench.py --model Qwen3-1.7B --with-cuda-graph
```

**v0–v3 横向对比**（在仓库根目录）：

```bash
python scripts/compare_v0_v3_bench.py --quick
python scripts/compare_v0_v3_bench.py --model Qwen3-0.6B --versions v0 v1 v2 v3
```

首次使用请先安装环境：

```bash
cd vllm-v0 && bash setup_env.sh && source .venv/bin/activate
python run.py --download
python run.py --verify-all
```

profile 与 bench 结果写入各版本 `bench/results/`。完整七版路线以上表为准，每完成一版即冻结该目录，在下一目录中增量开发。
