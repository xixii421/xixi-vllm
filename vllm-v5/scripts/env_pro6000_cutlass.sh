#!/usr/bin/env bash
# Blackwell (PRO 6000 / SM120) MoE cutlass + CUDA Graph env.
#
# System often has nvcc 12.8 only. Torch may report CUDA 12.9 and FlashInfer
# would pick arch 12.0f (needs nvcc≥12.9). We pin arch to 12.0a so cutlass JIT
# works with /usr/local/cuda-12.8.
#
# Usage:
#   source vllm-v7/scripts/env_pro6000_cutlass.sh
#   CUDA_VISIBLE_DEVICES=0 python ...

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

if [[ -d /usr/local/cuda-12.8 ]]; then
  export CUDA_HOME=/usr/local/cuda-12.8
elif [[ -d /usr/local/cuda ]]; then
  export CUDA_HOME=/usr/local/cuda
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

# Prefer a-family for SM120 when toolkit < 12.9 (cutlass still CG-capable).
# Do NOT set TORCH_CUDA_ARCH_LIST — that can break PyTorch built-in kernels on SM120.
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.0a}"

# Helpful defaults for MoE on PRO6000
export VLLM_V7_MOE_FUSED_BACKEND="${VLLM_V7_MOE_FUSED_BACKEND:-cutlass}"

# Custom moe CUDA aux (align/silu/sum) must include sm_120a — see cuda_ops.py.
# If you see "no kernel image" after changing arches, wipe:
#   rm -rf ~/.cache/torch_extensions/*/vllm_v7_moe_cuda

echo "[env_pro6000_cutlass] CUDA_HOME=${CUDA_HOME:-unset}"
echo "[env_pro6000_cutlass] FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}"
echo "[env_pro6000_cutlass] VLLM_V7_MOE_FUSED_BACKEND=${VLLM_V7_MOE_FUSED_BACKEND}"
command -v nvcc >/dev/null && nvcc --version | sed -n 's/^.*release /[env_pro6000_cutlass] nvcc release /p'
