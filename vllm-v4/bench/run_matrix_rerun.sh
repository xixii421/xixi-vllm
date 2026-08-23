#!/usr/bin/env bash
# 补跑失败配置：DP2（4B，16 KV 块适配 12GB 卡）+ 官方 vLLM（8B，gpu_memory_utilization=0.9）。
# 追加写 bench.jsonl，不清空。8b-pp2-compile 超时属已知限制，见文档。
set -u
cd "$(dirname "$0")/.." || exit 1

PY=../.venv2/bin/python
PY_VLLM=../.venv-vllm/bin/python
M8B="$HOME/shared/models/Qwen3-8B"
M4B="$HOME/huggingface/Qwen3-4B"

run_dp() {
  local dp=$1 tag=$2 cg=$3 compile=$4; shift 4
  local args=(--dp "$dp" --tag "$tag" "$@")
  [ "$cg" = 1 ] && args+=(--cg)
  [ "$compile" = 1 ] && args+=(--compile)
  echo "=== $tag ==="
  timeout 900 "$PY" bench/bench_dp.py "${args[@]}" 2>&1 | grep -E "tok/s" || echo "  (TIMEOUT/FAILED: $tag)"
}

run_vllm() {
  local tp=$1 tag=$2 cg=$3 compile=$4; shift 4
  local args=(--tp "$tp" --tag "$tag" "$@")
  [ "$cg" = 1 ] && args+=(--cg)
  [ "$compile" = 1 ] && args+=(--compile)
  echo "=== $tag ==="
  timeout 900 "$PY_VLLM" bench/bench_vllm_official.py "${args[@]}" 2>&1 | grep -E "^\S+ \|" || echo "  (TIMEOUT/FAILED: $tag)"
}

echo "########## v4: Qwen3-4B DP=2 (16 blocks) ##########"
run_dp 2 4b-dp2-eager      0 0 --model "$M4B" --num-kvcache-blocks 16
run_dp 2 4b-dp2-cg         1 0 --model "$M4B" --num-kvcache-blocks 16
run_dp 2 4b-dp2-compile    0 1 --model "$M4B" --num-kvcache-blocks 16
run_dp 2 4b-dp2-cgcompile  1 1 --model "$M4B" --num-kvcache-blocks 16

echo "########## official vLLM: Qwen3-8B ##########"
run_vllm 1 vllm-8b-tp1-eager    0 0 --model "$M8B" --cuda-devices 0
run_vllm 1 vllm-8b-tp1-cg       1 0 --model "$M8B" --cuda-devices 0
run_vllm 1 vllm-8b-tp1-compile  0 1 --model "$M8B" --cuda-devices 0
run_vllm 2 vllm-8b-tp2-eager    0 0 --model "$M8B" --cuda-devices 0,1
run_vllm 2 vllm-8b-tp2-cg       1 0 --model "$M8B" --cuda-devices 0,1
run_vllm 2 vllm-8b-tp2-compile  0 1 --model "$M8B" --cuda-devices 0,1

echo "=== RERUN DONE ==="
