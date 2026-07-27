#!/bin/bash
# 运行 vllm-v3 / ullm / official-vLLM 三路 prefill TTFT 对比
# 每个 backend 使用其最佳性能配置（enforce_eager=False）
#
# 用法:
#   cd /home/xingyongwang/code/mycuda/dzyy-vllm
#   bash vllm-v3/bench/run_prefill_3way.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH="$ROOT/vllm-v3/bench/bench_prefill_3way.py"
OUTDIR="$ROOT/vllm-v3/bench/out"
mkdir -p "$OUTDIR"

PYTHON_BASE=/home/xingyongwang/miniconda3/bin/python
PYTHON_VLLM="$ROOT/.venv-vllm/bin/python"

echo "============================================================"
echo "  Prefill TTFT 3-way Benchmark"
echo "  Model: Qwen3-0.6B @ RTX 4090D"
echo "  Backends: vllm-v3 | ullm | vLLM 0.26.0"
echo "  Config: enforce_eager=False (各自最高性能)"
echo "============================================================"
echo ""

# ── vllm-v3 ────────────────────────────────────────────────────
echo "[1/3] vllm-v3 (FlashInfer + CUDA Graph decode) ..."
PYTHONPATH="$ROOT/vllm-v3" CUDA_VISIBLE_DEVICES=1 $PYTHON_BASE "$BENCH" \
    --backend v3 --out "$OUTDIR/prefill_v3.json"

# ── ullm ────────────────────────────────────────────────────────
echo ""
echo "[2/3] ullm ..."
CUDA_VISIBLE_DEVICES=1 $PYTHON_BASE "$BENCH" \
    --backend ullm --out "$OUTDIR/prefill_ullm.json"

# ── official vLLM 0.26.0 ───────────────────────────────────────
echo ""
echo "[3/3] official vLLM 0.26.0 ..."
CUDA_VISIBLE_DEVICES=1 $PYTHON_VLLM "$BENCH" \
    --backend vllm --out "$OUTDIR/prefill_vllm.json"

# ── Compare ─────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  结果汇总"
echo "============================================================"

$PYTHON_BASE -c "
import json
from pathlib import Path

out = Path('$OUTDIR')
files = [
    ('v3',    'prefill_v3.json'),
    ('ullm',  'prefill_ullm.json'),
    ('vLLM',  'prefill_vllm.json'),
]

data = {}
for name, fn in files:
    fp = out / fn
    if not fp.exists():
        continue
    d = json.loads(fp.read_text())
    data[name] = d

# Table
names = list(data.keys())
header = f'{\"Prompt\":>10}'
for n in names:
    header += f'  {n:>10}'
print()
print(header)
print('-' * len(header))

prompts = sorted({k for d in data.values() for k in d if k != '_meta'})
for plen in prompts:
    row = f'{plen.replace(\"prefill_\", \"\"):>10}'
    for n in names:
        v = data[n].get(plen, {}).get('ttft_mean_ms', None)
        row += f'  {f\"{v:.1f}ms\" if v else \"N/A\":>10}'
    print(row)

# Summary: v3 vs competition
print()
baseline = 'v3'
if baseline in data:
    for n in names:
        if n == baseline:
            continue
        ratios = []
        for plen in prompts:
            b = data[baseline].get(plen, {}).get('ttft_mean_ms', 0)
            o = data[n].get(plen, {}).get('ttft_mean_ms', 0)
            if b > 0 and o > 0:
                ratios.append(o / b)
        if ratios:
            avg = sum(ratios) / len(ratios)
            label = 'SLOWER' if avg > 1.05 else ('FASTER' if avg < 0.95 else 'same')
            print(f'  {n} vs v3: {avg:.2%} ({label})')
"
