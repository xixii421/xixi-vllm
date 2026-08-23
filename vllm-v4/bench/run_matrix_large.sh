#!/usr/bin/env bash
# v4 大模型矩阵 + 官方 vLLM 对照。
# - v4:  8B 单卡/TP2/PP2、14B PP2(不均分 12,28)、4B DP1/DP2，各 × eager/CG/compile/CG+compile
# - vLLM: 8B 单卡(4090)/TP2 × eager/CG/compile（V1 无 PP；14B 本机无法跑）
# 结果写 bench/out/bench_parallel.txt / bench_dp.txt / bench_vllm_official.txt + bench.jsonl。
set -u
cd "$(dirname "$0")/.." || exit 1

PY=../.venv2/bin/python
PY_VLLM=../.venv-vllm/bin/python
M8B="$HOME/shared/models/Qwen3-8B"
M14B="$HOME/shared/models/Qwen3-14B"
M4B="$HOME/huggingface/Qwen3-4B"
JSONL=bench/out/bench.jsonl

: > "$JSONL"
: > bench/out/bench_parallel.txt
: > bench/out/bench_dp.txt
: > bench/out/bench_vllm_official.txt

# run_par <tp> <pp> <tag> <cg> <compile> [extra args...]
run_par() {
  local tp=$1 pp=$2 tag=$3 cg=$4 compile=$5; shift 5
  local args=(--tp "$tp" --pp "$pp" --tag "$tag" "$@")
  [ "$cg" = 1 ] && args+=(--cg)
  [ "$compile" = 1 ] && args+=(--compile)
  echo "=== $tag ==="
  timeout 900 "$PY" bench/bench_parallel.py "${args[@]}" 2>&1 | grep -E "^\S+ \|" || echo "  (TIMEOUT/FAILED: $tag)"
}

# run_dp <dp> <tag> <cg> <compile> [extra args...]
run_dp() {
  local dp=$1 tag=$2 cg=$3 compile=$4; shift 4
  local args=(--dp "$dp" --tag "$tag" "$@")
  [ "$cg" = 1 ] && args+=(--cg)
  [ "$compile" = 1 ] && args+=(--compile)
  echo "=== $tag ==="
  timeout 900 "$PY" bench/bench_dp.py "${args[@]}" 2>&1 | grep -E "tok/s" || echo "  (TIMEOUT/FAILED: $tag)"
}

# run_vllm <tp> <tag> <cg> <compile> [extra args...]
run_vllm() {
  local tp=$1 tag=$2 cg=$3 compile=$4; shift 4
  local args=(--tp "$tp" --tag "$tag" "$@")
  [ "$cg" = 1 ] && args+=(--cg)
  [ "$compile" = 1 ] && args+=(--compile)
  echo "=== $tag ==="
  timeout 900 "$PY_VLLM" bench/bench_vllm_official.py "${args[@]}" 2>&1 | grep -E "^\S+ \|" || echo "  (TIMEOUT/FAILED: $tag)"
}

echo "########## v4: Qwen3-8B ##########"
# 8B 单卡（cuda:0 = 4090 D）
run_par 1 1 8b-tp1-eager      0 0 --model "$M8B" --device-ids 0
run_par 1 1 8b-tp1-cg         1 0 --model "$M8B" --device-ids 0
run_par 1 1 8b-tp1-compile    0 1 --model "$M8B" --device-ids 0
run_par 1 1 8b-tp1-cgcompile  1 1 --model "$M8B" --device-ids 0
# 8B TP=2（rank0→cuda:0=4090, rank1→cuda:1=4070）
run_par 2 1 8b-tp2-eager      0 0 --model "$M8B"
run_par 2 1 8b-tp2-cg         1 0 --model "$M8B"
run_par 2 1 8b-tp2-compile    0 1 --model "$M8B"
run_par 2 1 8b-tp2-cgcompile  1 1 --model "$M8B"
# 8B PP=2（等分 18/18，rank0→4090, rank1→4070）
run_par 1 2 8b-pp2-eager      0 0 --model "$M8B"
run_par 1 2 8b-pp2-cg         1 0 --model "$M8B"
run_par 1 2 8b-pp2-compile    0 1 --model "$M8B"
run_par 1 2 8b-pp2-cgcompile  1 1 --model "$M8B"

echo "########## v4: Qwen3-14B (PP=2 不均分 12,28) ##########"
# 12 层→cuda:1=4070(12GB)，28 层→cuda:0=4090(24GB)：--device-ids 1,0
run_par 1 2 14b-pp2-eager      0 0 --model "$M14B" --pp-layer-counts 12,28 --device-ids 1,0 --num-kvcache-blocks 32
run_par 1 2 14b-pp2-cg         1 0 --model "$M14B" --pp-layer-counts 12,28 --device-ids 1,0 --num-kvcache-blocks 32
run_par 1 2 14b-pp2-compile    0 1 --model "$M14B" --pp-layer-counts 12,28 --device-ids 1,0 --num-kvcache-blocks 32
run_par 1 2 14b-pp2-cgcompile  1 1 --model "$M14B" --pp-layer-counts 12,28 --device-ids 1,0 --num-kvcache-blocks 32

echo "########## v4: Qwen3-4B (DP) ##########"
# DP=1 用 cuda:0=4090 做单卡基线；DP=2 用 0,1（replica0→4090, replica1→4070）
run_dp 1 4b-dp1-eager      0 0 --model "$M4B" --device-ids 0 --num-kvcache-blocks 48
run_dp 1 4b-dp1-cg         1 0 --model "$M4B" --device-ids 0 --num-kvcache-blocks 48
run_dp 1 4b-dp1-compile    0 1 --model "$M4B" --device-ids 0 --num-kvcache-blocks 48
run_dp 1 4b-dp1-cgcompile  1 1 --model "$M4B" --device-ids 0 --num-kvcache-blocks 48
run_dp 2 4b-dp2-eager      0 0 --model "$M4B" --num-kvcache-blocks 48
run_dp 2 4b-dp2-cg         1 0 --model "$M4B" --num-kvcache-blocks 48
run_dp 2 4b-dp2-compile    0 1 --model "$M4B" --num-kvcache-blocks 48
run_dp 2 4b-dp2-cgcompile  1 1 --model "$M4B" --num-kvcache-blocks 48

echo "########## official vLLM: Qwen3-8B ##########"
# CUDA_VISIBLE_DEVICES 用 torch 序：0=4090, 1=4070
run_vllm 1 vllm-8b-tp1-eager    0 0 --model "$M8B" --cuda-devices 0
run_vllm 1 vllm-8b-tp1-cg       1 0 --model "$M8B" --cuda-devices 0
run_vllm 1 vllm-8b-tp1-compile  0 1 --model "$M8B" --cuda-devices 0
run_vllm 2 vllm-8b-tp2-eager    0 0 --model "$M8B" --cuda-devices 0,1
run_vllm 2 vllm-8b-tp2-cg       1 0 --model "$M8B" --cuda-devices 0,1
run_vllm 2 vllm-8b-tp2-compile  0 1 --model "$M8B" --cuda-devices 0,1

echo "=== DONE ==="
