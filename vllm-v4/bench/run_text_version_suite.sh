#!/usr/bin/env bash
# Text-only head-to-head: v0 → v5 + official on one GPU.
#
# Usage:
#   bash vllm-v4/bench/run_text_version_suite.sh [GPU_ID]
#
# Env overrides:
#   MODEL=/path/to/Qwen3-0.6B
#   MAX_NEW=32
#   TAG=q06
#   GPU_UTIL=0.70          # lower if GPU is shared
#   VERSIONS="v0 v1 v2 v3 v4 v5"   # subset ok
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
V4="$(cd "$(dirname "$0")/.." && pwd)"
GPU="${1:-0}"
MODEL="${MODEL:-/mnt/nfs/zhangli/huggingface/Qwen3-0.6B}"
MAX_NEW="${MAX_NEW:-32}"
TAG="${TAG:-text}"
GPU_UTIL="${GPU_UTIL:-0.85}"
VERSIONS="${VERSIONS:-v0 v1 v2 v3 v4 v5}"
OUT_DIR="${V4}/bench/results/text_suite_${TAG}"
mkdir -p "$OUT_DIR"

echo "=== GPU $GPU text suite (v0–v5 + official) ==="
nvidia-smi -i "$GPU" --query-gpu=name,memory.free --format=csv
echo "model=$MODEL max_new=$MAX_NEW gpu_util=$GPU_UTIL versions=$VERSIONS"

source "$ROOT/.venv/bin/activate"
source "$ROOT/.env.cuda.sh"
export CUDA_VISIBLE_DEVICES="$GPU"

for ver in $VERSIONS; do
  echo ""
  echo "-------- $ver --------"
  SRC="$ROOT/vllm-$ver"
  if [[ ! -d "$SRC" ]]; then
    echo "WARN: missing $SRC, skip" >&2
    continue
  fi
  # v0/v1 have no continuous batching; keep serial conc small
  if [[ "$ver" == "v0" || "$ver" == "v1" ]]; then
    CONC="1,2"
  else
    CONC="1,2,4,8"
  fi
  PYTHONPATH="$SRC" python "$V4/bench/text_version_suite.py" \
    --version "$ver" \
    --model "$MODEL" \
    --max-new "$MAX_NEW" \
    --gpu-util "$GPU_UTIL" \
    --warmup 2 --iters 4 \
    --conc "$CONC" \
    --out "$OUT_DIR/${ver}.json" || echo "WARN: $ver failed" >&2
  python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
  sleep 2
done
deactivate || true

echo ""
echo "-------- official --------"
source "$ROOT/.venv-vllm/bin/activate"
source "$ROOT/.env.vllm.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
python "$V4/bench/text_version_suite.py" \
  --version official \
  --model "$MODEL" \
  --max-new "$MAX_NEW" \
  --gpu-util "$GPU_UTIL" \
  --warmup 2 --iters 4 \
  --conc 1,2,4,8 \
  --out "$OUT_DIR/official.json" || echo "WARN: official failed" >&2
deactivate || true

python3 - <<PY
import json
from pathlib import Path
d = Path("$OUT_DIR")
order = ["v0", "v1", "v2", "v3", "v4", "v5", "official"]
print("\\n======== TEXT SUITE VERDICT (v0–v5 + official) ========")
print(f"{'ver':10} {'load':>7} {'TTFT_ms':>8} {'dec_tok/s':>10} {'e2e_tok/s':>10} {'lat_ms':>8}  conc1/2/4/8")
for ver in order:
    p = d / f"{ver}.json"
    if not p.exists():
        print(f"{ver:10} MISSING")
        continue
    r = json.loads(p.read_text())
    s = r["single"]
    c = r["concurrency"]
    conc = " ".join(
        f"{c[f'n{n}']['batch_tok_s']:.0f}" for n in (1, 2, 4, 8) if f"n{n}" in c
    )
    print(
        f"{ver:10} {r['load_s']:7.1f} {s['ttft_s']*1000:8.1f} "
        f"{s['decode_tok_s']:10.1f} {s['e2e_tok_s']:10.1f} {s['e2e_latency_s']*1000:8.1f}  {conc}"
    )
print(f"\\nresults: {d}")
PY
