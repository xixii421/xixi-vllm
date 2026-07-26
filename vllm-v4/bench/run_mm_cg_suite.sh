#!/usr/bin/env bash
# Full CG multimodal suite: v4 vs official (same GPU / model / image scales / concurrency).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
V4="$(cd "$(dirname "$0")/.." && pwd)"
GPU="${1:-3}"
MODEL="${MODEL:-/mnt/nfs/zhangli/huggingface/Qwen3-VL-2B-Instruct}"
IMAGE="${IMAGE:-cat.jpg}"
MAX_NEW="${MAX_NEW:-32}"
WARMUP="${WARMUP:-1}"
ITERS="${ITERS:-3}"
SCALES="${SCALES:-256,512,768}"
CONC="${CONC:-1,2,4,8}"
OUT_DIR="${V4}/bench/results"
TAG="${TAG:-2b}"
mkdir -p "$OUT_DIR"

echo "=== GPU $GPU ==="
nvidia-smi -i "$GPU" --query-gpu=name,memory.free,utilization.gpu --format=csv || true

echo ""
echo "=== [1/2] vllm-v4 (decode CG + ViT CG) ==="
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
# shellcheck disable=SC1091
source "$ROOT/.env.cuda.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$V4"
PYTHONPATH=. python bench/mm_cg_suite_v4.py \
  --model "$MODEL" \
  --image "$IMAGE" \
  --max-new "$MAX_NEW" \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  --scales "$SCALES" \
  --conc "$CONC" \
  --gpu-util 0.85 \
  --out "$OUT_DIR/cg_suite_v4_${TAG}.json"
deactivate || true

echo ""
echo "=== [2/2] official vLLM (decode CG + encoder CG) ==="
# shellcheck disable=SC1091
source "$ROOT/.venv-vllm/bin/activate"
# shellcheck disable=SC1091
source "$ROOT/.env.vllm.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$V4"
python bench/mm_cg_suite_official.py \
  --model "$MODEL" \
  --image "$IMAGE" \
  --max-new "$MAX_NEW" \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  --scales "$SCALES" \
  --conc "$CONC" \
  --gpu-util 0.85 \
  --max-model-len 8192 \
  --out "$OUT_DIR/cg_suite_official_${TAG}.json"
deactivate || true

echo "Done: $OUT_DIR/cg_suite_*_${TAG}.json"
