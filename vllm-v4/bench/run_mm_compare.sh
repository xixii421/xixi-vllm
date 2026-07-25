#!/usr/bin/env bash
# Run multimodal stage compare without nvidia-smi / sudo.
# Usage (from repo root or vllm-v4):
#   bash bench/run_mm_compare.sh
# Optional:
#   MODEL=... IMAGE=... MAX_NEW=32 bash bench/run_mm_compare.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-/mnt/nfs/zhangli/huggingface/Qwen3-VL-8B-Instruct}"
IMAGE="${IMAGE:-$ROOT/cat.jpg}"
MAX_NEW="${MAX_NEW:-32}"
WARMUP="${WARMUP:-1}"
ITERS="${ITERS:-3}"
OUT_DIR="${OUT_DIR:-$ROOT/bench/results}"
mkdir -p "$OUT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# NVML is broken on this host; torch CUDA still works. Silence the warning noise.
export VLLM_DISABLE_NVML="${VLLM_DISABLE_NVML:-1}"

echo "==== host CUDA smoke (no nvidia-smi needed) ===="
"$REPO/.venv/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print(f"device={torch.cuda.get_device_name(0)} mem_gb={torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}")
x = torch.randn(256, 256, device="cuda"); y = x @ x; torch.cuda.synchronize()
print("matmul_ok")
PY

echo "==== [1/2] vllm-v4 (.venv) ===="
source "$REPO/.venv/bin/activate"
# shellcheck disable=SC1091
source "$REPO/.env.cuda.sh" || true
PYTHONPATH="$ROOT" python "$ROOT/bench/mm_compare_v4.py" \
  --model "$MODEL" --image "$IMAGE" --max-new "$MAX_NEW" \
  --warmup "$WARMUP" --iters "$ITERS" \
  --out "$OUT_DIR/v4_mm.json"

echo "==== [2/2] official vLLM (.venv-vllm) ===="
source "$REPO/.venv-vllm/bin/activate"
# shellcheck disable=SC1091
source "$REPO/.env.vllm.sh" || true
python "$ROOT/bench/mm_compare_official.py" \
  --model "$MODEL" --image "$IMAGE" --max-new "$MAX_NEW" \
  --warmup "$WARMUP" --iters "$ITERS" \
  --out "$OUT_DIR/official_mm.json"

echo "==== merge + report ===="
"$REPO/.venv/bin/python" "$ROOT/bench/mm_compare_report.py" \
  --v4 "$OUT_DIR/v4_mm.json" \
  --official "$OUT_DIR/official_mm.json" \
  --out "$OUT_DIR/mm_compare_report.md"

echo "Done. Report: $OUT_DIR/mm_compare_report.md"
