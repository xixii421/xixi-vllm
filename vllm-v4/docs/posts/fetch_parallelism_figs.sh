#!/usr/bin/env bash
# 重新获取 docs/并行环境.md 1.2 节的三张官方示意图。
# 这些图来自 Hugging Face《Model Parallelism》文档（v4.19.4），
# 其中 TP 图源自 Megatron-LM 论文，PP 图源自 GPipe 论文，DP 图源自
# DeepSpeed ZeRO 博客。下载后覆盖 docs/posts/ 下同名文件。
# 需要可访问 huggingface.co 的网络。若本地配置了代理，先 export
# http_proxy / https_proxy 再运行本脚本。
#
# 用法（在 vllm-v4/ 下）：
#     bash docs/posts/fetch_parallelism_figs.sh

set -euo pipefail
cd "$(dirname "$0")"

BASE="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main"

declare -A FILES=(
  ["parallelism_dp.png"]="parallelism-zero.png"                              # DP / ZeRO 数据并行
  ["parallelism_tp.png"]="parallelism-tp-parallel_shard_processing.png"      # TP，Megatron-LM 论文
  ["parallelism_pp.png"]="parallelism-gpipe-bubble.png"                      # PP，GPipe 论文
)

for out in "${!FILES[@]}"; do
  src="${FILES[$out]}"
  echo "fetching $src -> $out"
  curl -sfL --max-time 60 -o "$out" "$BASE/$src"
done

echo "done"
