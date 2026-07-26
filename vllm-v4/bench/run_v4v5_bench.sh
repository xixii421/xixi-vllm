#!/usr/bin/env bash
# Driver: run v4 and v5 benchmarks, merge results.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
V4_DIR="$BASE_DIR/vllm-v4"
V5_DIR="$BASE_DIR/vllm-v5"
ACTIVATE="$HOME/code/mycuda/dzyy-vllm/dzyy_vllm/.venv/bin/activate"
source "$ACTIVATE"

# Kill any previously running benchmark processes
pkill -f "run_single_version.py" 2>/dev/null || true
sleep 1

# Prevent torch.compile workers from reserving GPU memory
export TORCHINDUCTOR_COMPILE_WORKERS=0
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

MODE="${1:-text}"
mkdir -p "$SCRIPT_DIR/results"

echo "=== v4 ${MODE} ==="
OUT4="/tmp/v4_${MODE}.json"
PYTHONPATH="$V4_DIR" python3 "$SCRIPT_DIR/run_single_version.py" --version v4 --mode "$MODE" --out "$OUT4"

# Clean up between runs
pkill -f "run_single_version.py" 2>/dev/null || true
sleep 2
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
sleep 1

echo "=== v5 ${MODE} ==="
OUT5="/tmp/v5_${MODE}.json"
PYTHONPATH="$V5_DIR" python3 "$SCRIPT_DIR/run_single_version.py" --version v5 --mode "$MODE" --out "$OUT5"

# Merge into final result
OUT="$SCRIPT_DIR/results/v4v5_${MODE}.json"
python3 -c "
import json
with open('$OUT4') as f: v4 = json.load(f)
with open('$OUT5') as f: v5 = json.load(f)
combined = {'v4': v4, 'v5': v5}
with open('$OUT', 'w') as f: json.dump(combined, f, indent=2, default=str)
# Print comparison table
print()
print('='*70)
print('  COMPARISON TABLE')
print('='*70)
def print_row(a,b,c,d):
    print(f'| {a:<28s} | {b:>12s} | {c:>12s} | {d:>8s} |')
print_row('Metric','v4','v5','v4/v5')
print_row('-'*28,'-'*12,'-'*12,'-'*8)
for m in ('text','mm'):
    if m not in v4: continue
    if 'single' in v4.get(m,{}):
        for case in ['TTFT_1tok','short_32tok','med_128tok','long_256tok']:
            v4s = v4[m]['single'].get(case,{})
            v5s = v5[m]['single'].get(case,{})
            for k in ['ttft_ms','decode_tok_s','e2e_tok_s']:
                if k not in v4s or k not in v5s: continue
                v4v=float(v4s[k]); v5v=float(v5s[k])
                if v5v==0: continue
                ratio = v4v/v5v if 'tok_s' in k else v5v/v4v if v4v!=0 else 0
                print_row(f'{case}.{k}', f'{v4v:.2f}', f'{v5v:.2f}', f'{ratio:.2f}')
    if 'multimodal' in v4.get(m,{}):
        for case in v4[m]['multimodal']:
            v4s = v4[m]['multimodal'][case]
            v5s = v5[m]['multimodal'][case]
            for k in ['ttft_ms','decode_tok_s','e2e_tok_s']:
                if k not in v4s or k not in v5s: continue
                v4v=float(v4s[k]); v5v=float(v5s[k])
                if v5v==0: continue
                ratio = v4v/v5v if 'tok_s' in k else v5v/v4v if v4v!=0 else 0
                print_row(f'{case}.{k}', f'{v4v:.2f}', f'{v5v:.2f}', f'{ratio:.2f}')
    if 'concurrency' in v4.get(m,{}):
        for k in ['bs1','bs2','bs4','bs8']:
            if k not in v4[m]['concurrency']: continue
            v4s=v4[m]['concurrency'][k]; v5s=v5[m]['concurrency'][k]
            for sk in ['e2e_tok_s']:
                if sk not in v4s or sk not in v5s: continue
                v4v=float(v4s[sk]); v5v=float(v5s[sk])
                if v5v==0: continue
                print_row(f'concat.{k}.{sk}', f'{v4v:.2f}', f'{v5v:.2f}', f'{v4v/v5v:.2f}')
print(f'Saved to \$OUT')
"
