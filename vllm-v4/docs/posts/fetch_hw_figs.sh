#!/usr/bin/env bash
# 重新获取 docs/01-硬件互联.md 用到的官方示意图。
# 需要可访问外网。若本地配置了代理，先 export http_proxy / https_proxy。
#
# 用法（在 vllm-v4/ 下）：
#     bash docs/posts/fetch_hw_figs.sh

set -euo pipefail
cd "$(dirname "$0")"

curl -sfL --max-time 60 -o dgx_h100_gpu_tray.png \
  "https://docs.nvidia.com/dgx/dgxh100-user-guide/_images/dgx-h100-gpu-tray.png"
curl -sfL --max-time 60 -o hgx_h100_nvswitch.png \
  "https://developer-blogs.nvidia.com/wp-content/uploads/2022/03/HGX-H100-Diagram.png"
curl -sfL --max-time 60 -o hetero_straggler.png \
  "https://cdn.lablup.com/accl_heterogeneous_mix_en_1dddffa516.png"
curl -sfL --max-time 60 -o hetero_software_stacks.png \
  "https://cdn.lablup.com/diagram_e55875fa1a.png"
curl -sfL --max-time 60 -o hetero_cluster_pools.png \
  "https://cdn.lablup.com/diagram_2_68ec80fd2b.png"
curl -sfL --max-time 60 -o ascend_910_cluster.jpg \
  "https://www.servethehome.com/wp-content/uploads/2019/08/Huawei-Ascend-910-AI-Training-Cluster.jpg"

# §1.3 互联分档
curl -sfL --max-time 60 -o nvswitch_h100_labeled.png \
  "https://developer-blogs.nvidia.com/wp-content/uploads/2024/08/NVSwitch-image-fixed-1.png"
curl -sfL --max-time 60 -o nvlink_with_without_nvswitch.png \
  "https://developer-blogs.nvidia.com/wp-content/uploads/2024/08/gpu-to-gpu-bandwidth-nvswitch-comparison-b.png"
curl -sfL --max-time 60 -o nvlink_comms_nvswitch.png \
  "https://developer-blogs.nvidia.com/wp-content/uploads/2024/08/multi-gpu-communication-nvswitch-comparison.png"
curl -sfL --max-time 60 -o pcie_same_switch.jpg \
  "https://images.contentstack.io/v3/assets/blt71da4c740e00faaa/blt092ce52a86bd22a7/5eea92e18d39260a3ded6742/Figure-21.jpg"
curl -sfL --max-time 60 -o pcie_cross_numa.jpg \
  "https://images.contentstack.io/v3/assets/blt71da4c740e00faaa/blt0eb482732005c0c6/5eea92e1293eef25d2b6e298/Figure-31.jpg"
curl -sfL --max-time 60 -o gpudirect_host_bounce.png \
  "https://developer-blogs.nvidia.com/wp-content/uploads/2019/08/GPUDirect-Fig-1-New.png"

# 以下三张从 PDF 页裁出，无法直接 curl，已提交到仓库：
#   gpudirect_p2p.png      <- NVIDIA GPUDirect Technology Overview.pdf p.7
#                            https://developer.download.nvidia.com/devzone/devcenter/cuda/docs/GPUDirect_Technology_Overview.pdf
#   cxl_device_types.png   <- Sharma et al., ACM CSUR 2024 / arXiv:2306.11227 p.6
#   cxl_pooling.png        <- CXL Consortium, CXL 3.0 White Paper p.3
#   cxl_sharing.png        <- 同上

echo "done"
