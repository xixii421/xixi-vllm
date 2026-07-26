#!/usr/bin/env bash
# Multimodal suite: v4 + v5 + official (single / mixed / pure image / text).
#
# Usage:
#   bash vllm-v4/bench/run_mm_version_suite_local.sh [GPU_ID]
#
# Env:
#   MODEL=~/huggingface/Qwen3-VL-2B-Instruct
#   IMAGE=vllm-v4/cat.jpg
#   MAX_SIDE=768
#   MAX_NEW=128
#   TAG=vl2b
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
V4="$(cd "$(dirname "$0")/.." && pwd)"
GPU="${1:-1}"
MODEL="${MODEL:-$HOME/huggingface/Qwen3-VL-2B-Instruct}"
IMAGE="${IMAGE:-$V4/cat.jpg}"
MAX_SIDE="${MAX_SIDE:-768}"
MAX_NEW="${MAX_NEW:-128}"
TAG="${TAG:-vl2b}"
GPU_UTIL="${GPU_UTIL:-0.85}"
LEAN_VENV="${LEAN_VENV:-$ROOT/dzyy_vllm/.venv}"
OFFICIAL_VENV="${OFFICIAL_VENV:-$ROOT/.venv-vllm}"
OUT_DIR="${V4}/bench/results/mm_suite_${TAG}"
mkdir -p "$OUT_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"

echo "=== GPU $GPU MM suite (v4 / v5 / official) ==="
nvidia-smi -i "$GPU" --query-gpu=name,memory.free --format=csv
echo "model=$MODEL image=$IMAGE max_side=$MAX_SIDE max_new=$MAX_NEW"

run_lean() {
  local ver="$1"
  local src="$ROOT/vllm-$ver"
  echo ""
  echo "-------- $ver --------"
  # shellcheck disable=SC1090
  source "$LEAN_VENV/bin/activate"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="$GPU"
  PYTHONPATH="$src" python "$V4/bench/mm_version_suite.py" \
    --version "$ver" \
    --model "$MODEL" \
    --image "$IMAGE" \
    --max-side "$MAX_SIDE" \
    --max-new "$MAX_NEW" \
    --gpu-util "$GPU_UTIL" \
    --warmup 2 --iters 3 \
    --out "$OUT_DIR/${ver}.json" || echo "WARN: $ver failed" >&2
  python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
  deactivate || true
  sleep 2
}

run_lean v4
run_lean v5

echo ""
echo "-------- official --------"
# shellcheck disable=SC1090
source "$OFFICIAL_VENV/bin/activate"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python "$V4/bench/mm_version_suite.py" \
  --version official \
  --model "$MODEL" \
  --image "$IMAGE" \
  --max-side "$MAX_SIDE" \
  --max-new "$MAX_NEW" \
  --gpu-util "$GPU_UTIL" \
  --warmup 2 --iters 3 \
  --out "$OUT_DIR/official.json" || echo "WARN: official failed" >&2
deactivate || true

echo ""
echo "-------- module profile (v4) --------"
source "$LEAN_VENV/bin/activate"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHONPATH="$ROOT/vllm-v4" python "$V4/bench/mm_profile_modules.py" \
  --model "$MODEL" \
  --image "$IMAGE" \
  --max-side "$MAX_SIDE" \
  --max-new 64 \
  --iters 15 \
  --out "$OUT_DIR/v4_modules.json" || echo "WARN: module profile failed" >&2
deactivate || true

python3 - <<PY
import json
from pathlib import Path
d = Path("$OUT_DIR")
order = ["v4", "v5", "official"]
cases = ["single_image", "mixed", "pure_image", "text_only"]
print("\\n======== MM SUITE VERDICT ========")
print(f"{'case':14} {'ver':10} {'TTFT_ms':>8} {'vit_ms':>8} {'dec_tok/s':>10} {'e2e_tok/s':>10} {'lat_ms':>8} {'ptok':>6}")
for case in cases:
    for ver in order:
        p = d / f"{ver}.json"
        if not p.exists():
            print(f"{case:14} {ver:10} MISSING")
            continue
        r = json.loads(p.read_text())
        c = r.get("cases", {}).get(case)
        if not c:
            print(f"{case:14} {ver:10} NO_CASE")
            continue
        vit = c.get("vit_encode_s")
        vit_ms = f"{vit*1000:.1f}" if isinstance(vit, (int, float)) and vit else "-"
        print(
            f"{case:14} {ver:10} {c['ttft_s']*1000:8.1f} {vit_ms:>8} "
            f"{c['decode_tok_s']:10.1f} {c['e2e_tok_s']:10.1f} "
            f"{c['e2e_latency_s']*1000:8.1f} {c.get('prompt_tokens',0):6.0f}"
        )
    print("-" * 80)
print(f"results: {d}")
mod = d / "v4_modules.json"
if mod.exists():
    m = json.loads(mod.read_text())
    print("\\n======== v4 MODULE PROFILE ========")
    print(json.dumps(m.get("vit_attn", {}), indent=2))
    if "e2e_modules" in m:
        print(json.dumps(m["e2e_modules"], indent=2))
PY
