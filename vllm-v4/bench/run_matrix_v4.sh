#!/usr/bin/env bash
# v4 基准矩阵：单卡/TP2/PP2 × eager/CG/compile，逐个跑 bench_parallel.py。
# 结果由 bench_parallel.py 自身写入 bench/out/bench_parallel.txt。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=../.venv2/bin/python
OUT=bench/out/bench_parallel.txt
: > "$OUT"

run() {
  local tp=$1 pp=$2 tag=$3 cg=$4 compile=$5
  local args=(--tp "$tp" --pp "$pp" --tag "$tag")
  [ "$cg" = 1 ] && args+=(--cg)
  [ "$compile" = 1 ] && args+=(--compile)
  timeout 300 "$PY" bench/bench_parallel.py "${args[@]}" 2>&1 | grep -E "^\S+ \|"
}

# tp=1 pp=1 (单卡)
run 1 1 tp1pp1-eager       0 0
run 1 1 tp1pp1-cg          1 0
run 1 1 tp1pp1-compile     0 1
run 1 1 tp1pp1-cg-compile  1 1
# tp=2 pp=1
run 2 1 tp2pp1-eager       0 0
run 2 1 tp2pp1-cg          1 0
run 2 1 tp2pp1-compile     0 1
run 2 1 tp2pp1-cg-compile  1 1
# tp=1 pp=2
run 1 2 tp1pp2-eager       0 0
run 1 2 tp1pp2-cg          1 0
run 1 2 tp1pp2-compile     0 1
run 1 2 tp1pp2-cg-compile  1 1

echo "=== DONE ==="
