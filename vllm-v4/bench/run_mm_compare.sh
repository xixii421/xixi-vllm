#!/usr/bin/env bash
# Fair multimodal compare: vllm-v4 vs official vLLM (same GPU / image / max_new).
#
# Usage:
#   cd vllm-v4
#   bash bench/run_mm_compare.sh [GPU_ID]
#
# Requires:
#   ../.venv          + ../.env.cuda.sh     for v4
#   ../.venv-vllm     + ../.env.vllm.sh     for official
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"   # vllm-vx
V4="$(cd "$(dirname "$0")/.." && pwd)"        # vllm-v4
GPU="${1:-3}"
MODEL="${MODEL:-/mnt/nfs/zhangli/huggingface/Qwen3-VL-8B-Instruct}"
IMAGE="${IMAGE:-cat.jpg}"
MAX_SIDE="${MAX_SIDE:-512}"
MAX_NEW="${MAX_NEW:-32}"
WARMUP="${WARMUP:-1}"
ITERS="${ITERS:-3}"
OUT_DIR="${V4}/bench/results"
mkdir -p "$OUT_DIR"

echo "=== GPU $GPU check ==="
nvidia-smi -i "$GPU" --query-gpu=name,memory.free,utilization.gpu,temperature.gpu --format=csv

echo ""
echo "=== [1/3] vllm-v4 ==="
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
# shellcheck disable=SC1091
source "$ROOT/.env.cuda.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$V4"
PYTHONPATH=. python bench/mm_compare_v4.py \
  --model "$MODEL" \
  --image "$IMAGE" \
  --max-side "$MAX_SIDE" \
  --max-new "$MAX_NEW" \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  --gpu-util 0.85 \
  --out "$OUT_DIR/v4_mm.json"
deactivate || true

echo ""
echo "=== [2/3] official vLLM ==="
# shellcheck disable=SC1091
source "$ROOT/.venv-vllm/bin/activate"
# shellcheck disable=SC1091
source "$ROOT/.env.vllm.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$V4"
python bench/mm_compare_official.py \
  --model "$MODEL" \
  --image "$IMAGE" \
  --max-side "$MAX_SIDE" \
  --max-new "$MAX_NEW" \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  --gpu-util 0.85 \
  --max-model-len 8192 \
  --out "$OUT_DIR/official_mm.json"
deactivate || true

echo ""
echo "=== [3/3] merge report ==="
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
cd "$V4"
python bench/mm_compare_report.py \
  --v4 "$OUT_DIR/v4_mm.json" \
  --official "$OUT_DIR/official_mm.json" \
  --out "$OUT_DIR/mm_compare_report.md"
echo "Report: $OUT_DIR/mm_compare_report.md"
