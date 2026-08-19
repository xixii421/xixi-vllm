"""生成 docs/并行环境.md 的两张插图：
1. three_machines_topology.png  三台机器拓扑对照
2. collectives.png              集合通信原语示意
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / ".." / "docs" / "posts"

_FONT = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
fm.fontManager.addfont(_FONT)
plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False


def draw_topology() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), dpi=160)

    def gpu(ax, x, y, label, fill="#cfe8ff"):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.42, y - 0.30), 0.84, 0.60,
            boxstyle="round,pad=0.02", fc=fill, ec="#2b6cb0", lw=1.4))
        ax.text(x, y, label, ha="center", va="center", fontsize=9)

    def link(ax, x1, y1, x2, y2, label, color="#d14d4d"):
        ax.plot([x1, x2], [y1, y2], color=color, lw=2.2, zorder=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.16, label,
                ha="center", va="center", fontsize=8.5, color=color)

    def numa_zone(ax, xs, ys, label, w=3.05, h=1.05):
        x0, x1 = min(xs) - 0.7, max(xs) + 0.7
        y0, y1 = min(ys) - 0.45, max(ys) + 0.55
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02",
            fc="#f7fafc", ec="#a0aec0", lw=1.0, ls="--"))
        ax.text((x0 + x1) / 2, y1 - 0.1, label, ha="center", va="bottom",
                fontsize=8, color="#4a5568")

    # ---- 环境 A：4090D + 4070S，跨 NUMA，P2P NS ----
    ax = axes[0]
    ax.set_title("环境 A  本地工作站  2 卡强异构", fontsize=10)
    gpu(ax, 0, 1.3, "4090 D\n24GB")
    gpu(ax, 2.2, 1.3, "4070 SUPER\n12GB")
    link(ax, 0.45, 1.3, 1.75, 1.3, "SYS  无P2P", "#d14d4d")
    numa_zone(ax, [0], [1.3], "NUMA0")
    numa_zone(ax, [2.2], [1.3], "NUMA1")
    ax.text(1.1, -0.1, "跨 CPU UPI 总线，通信绕主机内存\n实测 AR 8KB 约 53us", ha="center",
            fontsize=7.5, color="#4a5568")
    ax.set_xlim(-1.2, 3.4); ax.set_ylim(-0.5, 2.2); ax.axis("off")

    # ---- 环境 B：4 卡，GPU0/1 PIX，GPU2/3 PXB，跨组 SYS，P2P OK ----
    ax = axes[1]
    ax.set_title("环境 B  四卡服务器  同构  P2P=OK", fontsize=10)
    for i, x in enumerate([0, 1.4, 3.6, 5.0]):
        gpu(ax, x, 1.3, f"GPU{i}")
    link(ax, 0.42, 1.3, 0.98, 1.3, "PIX", "#2b6cb0")
    link(ax, 4.02, 1.3, 4.58, 1.3, "PXB", "#2b6cb0")
    link(ax, 1.82, 1.45, 3.18, 1.45, "SYS", "#d14d4d")
    numa_zone(ax, [0, 1.4], [1.3], "NUMA0")
    numa_zone(ax, [3.6, 5.0], [1.3], "NUMA1")
    ax.text(2.5, -0.1, "GPU0+1 同交换机可做 TP=2\n跨 NUMA 组要避免", ha="center",
            fontsize=7.5, color="#4a5568")
    ax.set_xlim(-1.2, 6.2); ax.set_ylim(-0.5, 2.2); ax.axis("off")

    # ---- 环境 C：6 卡，GPU3 是 A100，其余 3090，P2P NS/CNS ----
    ax = axes[2]
    ax.set_title("环境 C  六卡服务器  A100+5x3090 强异构", fontsize=10)
    gpu(ax, 0, 1.3, "GPU0\n3090")
    gpu(ax, 1.4, 1.3, "GPU1\n3090")
    gpu(ax, 2.8, 1.3, "GPU2\n3090")
    gpu(ax, 4.6, 1.3, "GPU3\nA100 80G", "#ffe9c7")
    gpu(ax, 6.0, 1.3, "GPU4\n3090")
    gpu(ax, 7.4, 1.3, "GPU5\n3090")
    link(ax, 0.42, 1.3, 0.98, 1.3, "NODE", "#2b6cb0")
    link(ax, 1.82, 1.3, 2.38, 1.3, "PIX", "#2b6cb0")
    link(ax, 5.02, 1.3, 5.58, 1.3, "NODE", "#2b6cb0")
    link(ax, 6.42, 1.3, 6.98, 1.3, "PIX", "#2b6cb0")
    link(ax, 3.22, 1.45, 4.18, 1.45, "SYS", "#d14d4d")
    numa_zone(ax, [0, 1.4, 2.8], [1.3], "NUMA0")
    numa_zone(ax, [4.6, 6.0, 7.4], [1.3], "NUMA1")
    ax.text(3.7, -0.1, "异构不能同组 TP，A100 单独跑大模型\n3090 各自跑中小模型或做 PP", ha="center",
            fontsize=7.5, color="#4a5568")
    ax.set_xlim(-1.2, 8.6); ax.set_ylim(-0.5, 2.2); ax.axis("off")

    fig.suptitle("三台真实机器的卡间拓扑（本机实测 nvidia-smi topo）", fontsize=12, y=1.0)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "three_machines_topology.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("saved", OUT / "three_machines_topology.png")


def draw_collectives() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.0), dpi=160)
    BLUE = "#2b6cb0"
    RED = "#d14d4d"
    GRN = "#38a169"

    def nodes(ax, ys, labels):
        for x, (y, l) in enumerate(zip(ys, labels), start=1):
            ax.add_patch(mpatches.Circle((x, y), 0.38, fc="#cfe8ff",
                                         ec=BLUE, lw=1.4, zorder=3))
            ax.text(x, y, l, ha="center", va="center", fontsize=8, zorder=4)

    def arr(ax, x1, y1, x2, y2, color=GRN, lw=1.6):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    shrinkA=8, shrinkB=8), zorder=2)

    # ---- all-reduce ----
    ax = axes[0][0]
    ax.set_title("all-reduce  每卡贡献一段，全部拿到和", fontsize=9.5)
    nodes(ax, [1, 2, 3], ["A", "B", "C"])
    for i, y1 in enumerate([1, 2, 3], start=1):
        for y2 in [1, 2, 3]:
            if y1 != y2:
                arr(ax, i, y1, (i + 1) % 4 or 1, y2 if abs(y1 - y2) < 2 else 3)
    ax.text(2.5, 0.15, "结果 = A+B+C，每个节点都有完整和", ha="center", fontsize=8)
    ax.set_xlim(0.2, 3.8); ax.set_ylim(0, 3.8); ax.axis("off")

    # ---- all-gather ----
    ax = axes[0][1]
    ax.set_title("all-gather  每卡提供一段，拼接后全部拥有", fontsize=9.5)
    nodes(ax, [1, 2, 3], ["a", "b", "c"])
    for i, y in enumerate([1, 2, 3], start=1):
        arr(ax, i, y, i + 0.9, y, color=RED)
    for x in [2, 3]:
        pass
    ax.text(2.5, 0.15, "拼接 [a|b|c]，每卡持完整段", ha="center", fontsize=8)
    ax.set_xlim(0.2, 3.8); ax.set_ylim(0, 3.8); ax.axis("off")

    # ---- reduce-scatter ----
    ax = axes[1][0]
    ax.set_title("reduce-scatter  求和后按段分给各卡", fontsize=9.5)
    nodes(ax, [1, 2, 3], ["a", "b", "c"])
    arr(ax, 1, 1, 1, 3, color=BLUE)
    arr(ax, 2, 2, 2, 3, color=BLUE)
    arr(ax, 1, 1, 1, 1, color=BLUE)
    ax.text(2.5, 0.15, "卡0 得 Σa，卡1 得 Σb，卡2 得 Σc", ha="center", fontsize=8)
    ax.set_xlim(0.2, 3.8); ax.set_ylim(0, 3.8); ax.axis("off")

    # ---- broadcast ----
    ax = axes[1][1]
    ax.set_title("broadcast  一张卡发给组内所有卡", fontsize=9.5)
    nodes(ax, [1, 2, 3], ["src", "dst", "dst"])
    arr(ax, 1, 2, 1, 1, color=RED)
    arr(ax, 1, 2, 1, 3, color=RED)
    ax.text(2.5, 0.15, "源卡数据复制到所有卡", ha="center", fontsize=8)
    ax.set_xlim(0.2, 3.8); ax.set_ylim(0, 3.8); ax.axis("off")

    fig.suptitle("集合通信四种原语的数据流", fontsize=12)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "collectives.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT / "collectives.png")


if __name__ == "__main__":
    draw_topology()
    draw_collectives()
