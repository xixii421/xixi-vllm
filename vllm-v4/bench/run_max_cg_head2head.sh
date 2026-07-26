#!/usr/bin/env bash
# Max-CG head-to-head on one GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
V4="$(cd "$(dirname "$0")/.." && pwd)"
GPU="${1:-3}"
MODEL="${MODEL:-/mnt/nfs/zhangli/huggingface/Qwen3-VL-2B-Instruct}"
IMAGE="${IMAGE:-cat.jpg}"
MAX_SIDE="${MAX_SIDE:-512}"
MAX_NEW="${MAX_NEW:-64}"
OUT_DIR="${V4}/bench/results"
TAG="${TAG:-maxcg}"
mkdir -p "$OUT_DIR"

echo "=== GPU $GPU max-CG head2head ==="
nvidia-smi -i "$GPU" --query-gpu=name,memory.free --format=csv

source "$ROOT/.venv/bin/activate"
source "$ROOT/.env.cuda.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$V4"
PYTHONPATH=. python bench/mm_max_cg_head2head.py --backend v4 \
  --model "$MODEL" --image "$IMAGE" --max-side "$MAX_SIDE" --max-new "$MAX_NEW" \
  --gpu-util 0.90 --vit-max-batch 8 --vit-budgets 64,128,256,512,1024,2048 \
  --warmup 2 --iters 5 --conc 1,2,4,8 \
  --out "$OUT_DIR/maxcg_v4_${TAG}.json"
deactivate || true

source "$ROOT/.venv-vllm/bin/activate"
source "$ROOT/.env.vllm.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$V4"
python bench/mm_max_cg_head2head.py --backend official \
  --model "$MODEL" --image "$IMAGE" --max-side "$MAX_SIDE" --max-new "$MAX_NEW" \
  --gpu-util 0.90 --vit-max-batch 8 --vit-budgets 64,128,256,512,1024,2048 \
  --warmup 2 --iters 5 --conc 1,2,4,8 \
  --out "$OUT_DIR/maxcg_official_${TAG}.json"
deactivate || true

python3 - <<PY
import json
from pathlib import Path
v=json.loads(Path("$OUT_DIR/maxcg_v4_${TAG}.json").read_text())
o=json.loads(Path("$OUT_DIR/maxcg_official_${TAG}.json").read_text())
print("\\n======== MAX-CG VERDICT ========")
print(f"load:     v4 {v['load_s']:.1f}s   official {o['load_s']:.1f}s")
for k in ("text_only","mm_image"):
    a,b=v["cases"][k], o["cases"][k]
    print(f"{k}:")
    print(f"  TTFT       v4 {a['ttft_s']*1000:.1f}ms  official {b['ttft_s']*1000:.1f}ms")
    if a.get("vit_encode_s") is not None:
        print(f"  ViT        v4 {a['vit_encode_s']*1000:.1f}ms  (official n/a)")
        print(f"  LM prefill v4 {a['lm_prefill_s']*1000:.1f}ms  (official folded into TTFT)")
    print(f"  decode     v4 {a['decode_tok_s']:.1f}   official {b['decode_tok_s']:.1f} tok/s")
    print(f"  e2e        v4 {a['e2e_tok_s']:.1f}   official {b['e2e_tok_s']:.1f} tok/s")
for n in sorted(v["concurrency"]):
    print(f"conc {n}: v4 {v['concurrency'][n]['batch_tok_s']:.1f}  official {o['concurrency'][n]['batch_tok_s']:.1f} tok/s")
PY
