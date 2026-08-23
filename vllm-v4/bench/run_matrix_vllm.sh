#!/usr/bin/env bash
# 官方 vLLM 基准矩阵：tp=1/2 × eager/CG/compile。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=../.venv-vllm/bin/python
OUT=bench/out/bench_vllm_official.txt
: > "$OUT"

run() {
  local tp=$1 tag=$2 cg=$3 compile=$4
  local args=(--tp "$tp" --tag "$tag")
  [ "$cg" = 1 ] && args+=(--cg)
  [ "$compile" = 1 ] && args+=(--compile)
  timeout 600 "$PY" bench/bench_vllm_official.py "${args[@]}" 2>&1 | grep -E "^\S+ \|"
}

run 1 vllm-tp1-eager      0 0
run 1 vllm-tp1-cg         1 0
run 1 vllm-tp1-compile    0 1
run 2 vllm-tp2-eager      0 0
run 2 vllm-tp2-cg         1 0
run 2 vllm-tp2-compile    0 1

echo "=== DONE ==="
