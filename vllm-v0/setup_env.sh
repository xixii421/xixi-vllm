#!/bin/bash
# vllm-v0 uv 环境安装
# 用法: bash setup_env.sh

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v uv &> /dev/null; then
    echo "[ERROR] 未找到 uv，请先安装: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "[INFO] $(uv --version)"

detect_cuda() {
    local DRIVER_VER
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    case "$DRIVER_VER" in
        545|546|547|548|549|550|551|552|553|554|555) echo "124" ;;
        560|565|570|575|580|585|590|595)             echo "126" ;;
        *) echo "121" ;;
    esac
}

CUDA_VER=$(detect_cuda)
echo "[INFO] CUDA wheel index: cu${CUDA_VER}"

if [ ! -d ".venv" ]; then
    echo "[INFO] 创建 uv 虚拟环境 .venv ..."
    uv venv .venv --python 3.11
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[INFO] 安装 PyTorch (cu${CUDA_VER}) ..."
uv pip install torch==2.5.1 \
    --index-url "https://download.pytorch.org/whl/cu${CUDA_VER}" \
    --extra-index-url https://pypi.org/simple

echo "[INFO] 安装 vllm-v0 依赖 ..."
uv pip install -r requirements.txt

echo ""
echo "=========================================="
python -c "
import torch
from transformers import __version__ as tf_ver
print(f'PyTorch:       {torch.__version__} (CUDA {torch.version.cuda})')
print(f'transformers:  {tf_ver}')
if torch.cuda.is_available():
    print(f'GPU:           {torch.cuda.get_device_name(0)}')
    print(f'VRAM:          {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('GPU:           not available')
print()
print('环境就绪')
print('  单模型推理:  python run.py')
print('  多模型验证:  python run.py --verify-all')
print('  仅下载权重:  python run.py --download')
"
