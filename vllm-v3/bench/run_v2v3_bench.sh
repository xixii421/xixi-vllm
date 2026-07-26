#!/usr/bin/env bash
# Orchestrate v2 vs v3 benchmarks across models and configurations.
# Each version runs in a subprocess with its own PYTHONPATH.

set -euo pipefail

REPO_DIR="/home/xingyongwang/code/mycuda/dzyy-vllm"
VENV="$REPO_DIR/dzyy_vllm/.venv/bin/activate"
MODEL_DIR="/home/xingyongwang/huggingface"
OUT_DIR="/tmp/v2v3_bench"

mkdir -p "$OUT_DIR"

MODELS=("Qwen3-0.6B" "Qwen3-1.7B" "Qwen3-4B")

# ── Functions ──

clean_gpu() {
    pkill -f "compile_worker" 2>/dev/null || true
    source "$VENV"
    python -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize()" 2>/dev/null
}

bench_config() {
    local label="$1"
    local pypath="$2"
    local model="$3"
    local model_path="$MODEL_DIR/$model"
    local out="$OUT_DIR/${label}_${model}.json"

    if [ ! -d "$model_path" ]; then
        echo "  SKIP $label $model (not found)"
        return
    fi

    echo ""
    echo ">>> BENCH: $label  model=$model"
    clean_gpu

    TORCHINDUCTOR_COMPILE_WORKERS=0 \
    PYTHONPATH="$REPO_DIR/$pypath" \
    "$REPO_DIR/dzyy_vllm/.venv/bin/python" \
        "$REPO_DIR/vllm-v3/bench/bench_v2v3.py" \
        --version "$label" --model "$model_path" --out "$out" \
        2>&1

    echo "<<< DONE: $label $model"
}

# ── Run benchmarks ──

for model in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "  MODEL: $model"
    echo "========================================"

    # v2 eager
    bench_config "v2_eager" "vllm-v2" "$model"

    # v2 compile
    bench_config "v2_compile" "vllm-v2" "$model"

    # v3 eager
    bench_config "v3_eager" "vllm-v3" "$model"

    # v3 cudagraph
    bench_config "v3_cudagraph" "vllm-v3" "$model"
done

echo ""
echo "========================================"
echo "  ALL BENCHMARKS COMPLETE"
echo "========================================"
echo "Results in: $OUT_DIR/"
ls -la "$OUT_DIR/"*.json 2>/dev/null || echo "(no results)"
