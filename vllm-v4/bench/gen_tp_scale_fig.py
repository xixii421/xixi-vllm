"""生成 TP 正负收益扫描热力图（tp2/tp1 单步比值，<1 为 TP 正收益）。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / ".." / "docs" / "posts"

_FONT = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
fm.fontManager.addfont(_FONT)
plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False

# (模型, bs) -> tp2/tp1，来自 bench_tp_scale.py rank0(4090D) 与 rank1(4070S)
MODELS = ["0.6B", "1.7B", "4B", "8B", "14B"]
BS = ["1", "16", "64", "128"]

RANK0 = np.array([
    [3.57, 3.52, 3.19, 3.70],
    [2.90, 3.46, 3.86, 6.25],
    [2.35, 1.57, 2.87, 4.04],
    [2.19, 2.61, 2.92, 3.79],
    [2.17, 2.45, 2.83, 3.01],
])
RANK1 = np.array([
    [3.35, 3.00, 2.48, 1.98],
    [0.73, 0.72, 0.96, 1.27],
    [0.59, 0.42, 0.76, 1.02],
    [0.54, 0.65, 0.68, 0.84],
    [0.52, 0.58, 0.68, 0.76],
])


def _draw(ax, data, title):
    vmin, vmax = 0.4, 4.2
    cmap = plt.cm.RdYlGn_r  # 红 = 大比值（负收益），绿 = 小比值（正收益）
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(BS)))
    ax.set_xticklabels([f"bs={b}" for b in BS])
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_title(title, fontsize=11)
    for i in range(len(MODELS)):
        for j in range(len(BS)):
            v = data[i, j]
            color = "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color=color)
    return im


fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), dpi=160)
im = _draw(axes[0], RANK0,
           "rank0  RTX 4090 D   tp2/tp1（>1 TP 负收益）")
_draw(axes[1], RANK1,
      "rank1  RTX 4070 SUPER   tp2/tp1（<1 TP 正收益）")
cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04)
cb.set_label("tp2 单步 / tp1 单步")
fig.suptitle("Qwen3 系列  TP 正负收益扫描（本机无 NVLink，NCCL 经 PCIe）",
             fontsize=12, y=1.02)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "tp_scale_heatmap.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved", OUT / "tp_scale_heatmap.png")
