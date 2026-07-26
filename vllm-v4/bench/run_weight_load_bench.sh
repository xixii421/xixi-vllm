#!/usr/bin/env bash
# Run weight-load benchmarks sequentially across all version×model combos.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LEAN_VENV="${LEAN_VENV:-$ROOT/dzyy_vllm/.venv}"
OFFICIAL_VENV="${OFFICIAL_VENV:-$ROOT/.venv-vllm}"
HF="${HF:-$HOME/huggingface}"
ITERS=5
SCRIPT="$ROOT/vllm-v4/bench/bench_weight_load.py"
OUT="$ROOT/vllm-v4/bench/results"
mkdir -p "$OUT"

bench() {
    local version=$1 model=$2 outfile=$3
    echo ""
    echo "========== $version  $(basename $model) =========="
    if [[ "$version" == "official" ]]; then
        source "$OFFICIAL_VENV/bin/activate"
        cd "$ROOT"
        PYTHONPATH="" python "$SCRIPT" --version official --model "$model" \
            --iters "$ITERS" --out "$OUT/$outfile"
    elif [[ "$version" == "v4" ]]; then
        source "$LEAN_VENV/bin/activate"
        PYTHONPATH="$ROOT/vllm-v4" python "$SCRIPT" --version v4 --model "$model" \
            --iters "$ITERS" --out "$OUT/$outfile"
    elif [[ "$version" == "v5" ]]; then
        source "$LEAN_VENV/bin/activate"
        PYTHONPATH="$ROOT/vllm-v5" python "$SCRIPT" --version v5 --model "$model" \
            --iters "$ITERS" --out "$OUT/$outfile"
    fi
    echo ""
}

# ── text models ────────────────────────────────────────────────
bench v4       "$HF/Qwen3-0.6B"           wl_v4_text.json
bench v5       "$HF/Qwen3-0.6B"           wl_v5_text.json
bench official "$HF/Qwen3-0.6B"           wl_off_text.json

# ── multimodal models ──────────────────────────────────────────
bench v4       "$HF/Qwen3-VL-2B-Instruct" wl_v4_vl.json
bench v5       "$HF/Qwen3-VL-2B-Instruct" wl_v5_vl.json
bench official "$HF/Qwen3-VL-2B-Instruct" wl_off_vl.json

# ── assemble verdict table ─────────────────────────────────────
echo ""
echo "================================================"
echo "  Verdict Table"
echo "================================================"
cd "$OUT" && python3 -c "
import json, glob

data = {f: json.load(open(f)) for f in glob.glob('wl_*.json')}

models = [
    ('Qwen3-0.6B',     [d for d in data.values() if not d['is_multimodal']]),
    ('Qwen3-VL-2B',    [d for d in data.values() if d['is_multimodal']]),
]

print()
print('┌' + '─'*108 + '┐')
print('│  Model weight loading speed  (disk → GPU, 5 iters, RTX 4090 D)')
print('├' + '─'*108 + '┤')
hdr = '│ {:<15s} {:<9s} {:>7s} {:>9s} {:>8s} {:>8s} {:>9s} {:>11s} {:>8s} │'
sep = '│' + '─'*15 + '┬' + '─'*9 + '┬' + '─'*7 + '┬' + '─'*9 + '┬' + '─'*8 + '┬' + '─'*8 + '┬' + '─'*9 + '┬' + '─'*11 + '┬' + '─'*8 + '│'
print(hdr.format('model','backend','disk','total_mean','create','load','weight_mean','ratio(total)',''))  # noqa
print('├' + '─'*15 + '┼' + '─'*9 + '┼' + '─'*7 + '┼' + '─'*9 + '┼' + '─'*8 + '┼' + '─'*8 + '┼' + '─'*9 + '┼' + '─'*11 + '┼' + '─'*8 + '┤')

fmt_data = '│ {:<15s} {:<9s} {:>5.1f}G {:>7.3f}s {:>6.3f}s {:>6.3f}s {:>7.3f}s {:>9.1f}% {:>6s} │'

for label, group in models:
    off = [d for d in group if d['backend']=='official']
    off_med = off[0]['total_median_s'] if off else 1.0
    for d in sorted(group, key=lambda x: x['backend']):
        b = d['backend']
        ds = d['disk_size_gb']
        tm = d['total_mean_s']
        cm = d['create_mean_s']
        lm = d['weight_mean_s']
        wm = d['weight_mean_s']
        ratio = (d['total_median_s'] / off_med * 100) if off_med > 0 else 0
        print(fmt_data.format(label, b, ds, tm, cm, lm, wm, ratio, ''))
print('├' + '─'*15 + '┴' + '─'*9 + '┴' + '─'*7 + '┴' + '─'*9 + '┴' + '─'*8 + '┴' + '─'*8 + '┴' + '─'*9 + '┴' + '─'*11 + '┴' + '─'*8 + '┤')
print('│  total=model_creation + weight_load | ratio = total_median / official_median * 100% │')
print('│  create: model __init__+to(device)  | load: disk read + CPU→GPU copy               │')
print('└' + '─'*108 + '┘')
"
