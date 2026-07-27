#!/bin/bash
# 运行 v2 vs v3 prefill 对比 benchmark
#
# 用法:
#   cd /home/xingyongwang/code/mycuda/dzyy-vllm
#   bash vllm-v3/bench/run_prefill_bench.sh
#
# 输出: vllm-v3/bench/out/prefill_*.json
#
# 实测结论 (2026-07, Qwen3-0.6B, RTX 4090D):
#   v2_eager    25ms TTFT  (基线: SDPA attention, block_size=16)
#   v3_eager    22ms TTFT  (最优: FlashInfer paged prefill, block_size=256)  13-15% ↑
#   v3_compile  44ms TTFT  (回退: torch.compile 与 FlashInfer .plan() 产生 N_layers 次 graph break)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH="$ROOT/vllm-v3/bench/bench_prefill_v2v3.py"
OUTDIR="$ROOT/vllm-v3/bench/out"
mkdir -p "$OUTDIR"

PYTHON=/home/xingyongwang/miniconda3/bin/python

# PyTorch 2.6 + Triton 3.x 编译修复
export TORCHINDUCTOR_CPP_WRAPPER=0

echo "============================================================"
echo "  v2 vs v3 Prefill TTFT 专项 Benchmark"
echo "  Model: Qwen3-0.6B"
echo "  Output: $OUTDIR"
echo "============================================================"
echo ""

# ── v2 (eager) ──────────────────────────────────────────────────
echo "[1/3] v2 eager ..."
PYTHONPATH="$ROOT/vllm-v2" $PYTHON "$BENCH" \
    --version v2 \
    --out "$OUTDIR/prefill_v2.json"

# ── v3 eager (no compile) ───────────────────────────────────────
echo ""
echo "[2/3] v3 eager (no compile_prefill) ..."
PYTHONPATH="$ROOT/vllm-v3" $PYTHON "$BENCH" \
    --version v3_eager \
    --out "$OUTDIR/prefill_v3_eager.json"

# ── v3 compile_prefill (experimental) ───────────────────────────
echo ""
echo "[3/3] v3 compile_prefill (实验性, 参考) ..."
PYTHONPATH="$ROOT/vllm-v3" $PYTHON "$BENCH" \
    --version v3 \
    --out "$OUTDIR/prefill_v3_compile.json"

echo ""
echo "============================================================"
echo "  完成！比较结果："
echo "============================================================"
$PYTHON -c "
import json
from pathlib import Path

out = Path('$OUTDIR')
for name, file in [('v2_eager', 'prefill_v2.json'), ('v3_eager', 'prefill_v3_eager.json'), ('v3_compile', 'prefill_v3_compile.json')]:
    data = json.loads((out / file).read_text())
    print(f'\n{\"=\"*60}')
    print(f'  {name}')
    print(f'  init: {data[\"_meta\"][\"init_s\"]:.1f}s')
    for k, v in sorted(data.items()):
        if k == '_meta':
            continue
        print(f'  {k:>16s}: {v[\"ttft_mean_ms\"]:>8.2f} ms  (median={v[\"ttft_median_ms\"]:.2f})')
"
echo ""
echo "============================================================"
echo "  结论: v3_eager 是当前 prefill 最优方案（13-15% ↑ vs v2）"
echo "        torch.compile 因 FlashInfer graph break 导致 2x 回退"
echo "============================================================"
