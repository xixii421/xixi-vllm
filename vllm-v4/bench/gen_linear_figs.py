"""生成线性层文档的结构示意图。

输出到 docs/posts/：
    linear_shard.png        列并行与行并行的切分对照
    mlp_linear_graph.png    Qwen3 解码层内的线性层连接与通信原语
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / ".." / "docs" / "posts"

_FONT = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
fm.fontManager.addfont(_FONT)
plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False

C_MAIN = "#2563eb"
C_HL = "#f59e0b"
C_DIM = "#9ca3af"
C_BG = "#f8fafc"
C_GREEN = "#16a34a"


def _box(ax, x, y, w, h, text, fc=C_BG, ec=C_MAIN, fs=10, tc="black",
         lw=1.4, rounded=0.05):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounded}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3)


def _arrow(ax, p0, p1, color=C_MAIN, lw=1.6, style="-|>", shrink=1.0,
           curved=None):
    if curved:
        ar = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                             linewidth=lw, color=color, zorder=4,
                             shrinkA=shrink, shrinkB=shrink,
                             connectionstyle=f"arc3,rad={curved}")
    else:
        ar = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                             linewidth=lw, color=color, zorder=4,
                             shrinkA=shrink, shrinkB=shrink)
    ax.add_patch(ar)


def _matrix(ax, x, y, rows, cols, cell_w, cell_h, fc="#f1f5f9",
            ec="#cbd5e1", label=None, hl_rows=(), hl_cols=(), fs=8.5):
    for r in range(rows):
        for c in range(cols):
            rx = x + c * cell_w
            ry = y + (rows - 1 - r) * cell_h
            hl = any(lo <= r < hi for lo, hi in hl_rows) or \
                 any(lo <= c < hi for lo, hi in hl_cols)
            _box(ax, rx, ry, cell_w, cell_h, "",
                 fc="#fef3c7" if hl else fc,
                 ec=C_HL if hl else ec, lw=0.8)
    if label:
        _box(ax, x, y + rows * cell_h + 0.08, cols * cell_w, 0.34,
             label, fc="#dbeafe", ec=C_MAIN, fs=fs)


def draw_linear_shard() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 6.6), dpi=160)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    # ================= 上子图 列并行 =================
    ax.text(0.3, 6.15, "列并行  按输出维切分，输出 all-gather 拼接",
            fontsize=10.5, color=C_MAIN, fontweight="bold")
    # 输入 x
    _box(ax, 0.3, 5.0, 1.15, 0.6, "输入 x\n[1, in]", fc="#e0f2fe", fs=8.5)
    # 完整权重 W [6,4]（上 3 行橙、下 3 行绿）
    wx, wy = 1.95, 3.9
    _matrix(ax, wx, wy, 6, 4, 0.5, 0.36, hl_rows=((0, 3), (3, 6)))
    _box(ax, wx - 0.06, wy + 6 * 0.36 + 0.08, 4 * 0.5 + 0.12, 0.34,
         "完整权重 W [out, in]", fc="#dbeafe", ec=C_MAIN, fs=8.5)
    _arrow(ax, (1.45, 5.25), (wx, 5.05), color=C_MAIN, lw=1.6)
    # 两块局部权重
    _matrix(ax, 4.7, 4.0, 3, 4, 0.5, 0.36, fc="#fef3c7", ec=C_HL,
            label="卡 0 权重 [out/tp, in]")
    _matrix(ax, 7.2, 4.0, 3, 4, 0.5, 0.36, fc="#ecfdf5", ec=C_GREEN,
            label="卡 1 权重 [out/tp, in]")
    _arrow(ax, (wx + 2.0, 5.1), (4.7, 5.05), color=C_HL, lw=1.5)
    _arrow(ax, (wx + 2.0, 4.15), (7.2, 4.15), color=C_GREEN, lw=1.5)
    # 局部输出
    _box(ax, 10.15, 4.85, 1.2, 0.5, "卡 0 输出\n[1, out/tp]", fc="#fef3c7",
         fs=8)
    _box(ax, 10.15, 3.85, 1.2, 0.5, "卡 1 输出\n[1, out/tp]", fc="#ecfdf5",
         fs=8)
    _arrow(ax, (9.7, 5.1), (10.15, 5.1), color=C_HL, lw=1.5)
    _arrow(ax, (9.7, 4.1), (10.15, 4.1), color=C_GREEN, lw=1.5)
    # all-gather
    _box(ax, 11.75, 4.2, 1.4, 0.85, "all-gather\n沿最后一维拼接",
         fc="#fee2e2", ec="#dc2626", fs=8)
    _arrow(ax, (11.35, 5.1), (11.75, 4.85), color=C_MAIN, lw=1.5)
    _arrow(ax, (11.35, 4.1), (11.75, 4.4), color=C_MAIN, lw=1.5)
    _box(ax, 11.95, 5.15, 1.15, 0.4, "完整输出\n[1, out]", fc="#fce7f3",
         fs=8.5)
    _arrow(ax, (12.45, 5.05), (12.45, 5.15), color=C_MAIN, lw=1.5)
    ax.text(0.3, 3.75, "每卡只算自己那一段，输出按序拼回完整向量，"
            "数学上等价于一次大矩阵乘",
            fontsize=9, color="#334155")

    # ================= 下子图 行并行 =================
    ax.text(0.3, 3.0, "行并行  按输入维切分，输出 all-reduce 求和",
            fontsize=10.5, color=C_GREEN, fontweight="bold")
    # 完整输入横条 [1, 6]，左 3 格橙、右 3 格绿
    _matrix(ax, 0.3, 1.95, 1, 6, 0.5, 0.4,
            hl_rows=((0, 0),), hl_cols=((0, 3), (3, 6)))
    _box(ax, 0.3, 2.45, 3.0, 0.34, "输入 x [1, in] 切成两段", fc="#dbeafe",
         ec=C_MAIN, fs=8.5)
    ax.text(0.55, 2.17, "x0", fontsize=9, color=C_HL, fontweight="bold")
    ax.text(2.05, 2.17, "x1", fontsize=9, color=C_GREEN, fontweight="bold")
    # 完整权重 W [4, 6]，左 3 列橙、右 3 列绿
    _matrix(ax, 4.0, 0.75, 4, 6, 0.42, 0.34,
            hl_cols=((0, 3), (3, 6)))
    _box(ax, 4.0, 2.25, 2.52, 0.34, "完整权重 W 沿输入维切", fc="#dbeafe",
         ec=C_MAIN, fs=8.5)
    ax.text(4.25, 1.9, "W0", fontsize=9, color=C_HL, fontweight="bold")
    ax.text(5.15, 1.9, "W1", fontsize=9, color=C_GREEN, fontweight="bold")
    # 部分和
    _box(ax, 0.4, 0.5, 1.7, 0.5, "卡 0 部分和 x0·W0\n[1, out]", fc="#fef3c7",
         fs=8)
    _box(ax, 2.5, 0.5, 1.7, 0.5, "卡 1 部分和 x1·W1\n[1, out]", fc="#ecfdf5",
         fs=8)
    _arrow(ax, (1.8, 1.95), (1.25, 1.0), color=C_HL, lw=1.5)
    _arrow(ax, (3.3, 1.95), (3.35, 1.0), color=C_GREEN, lw=1.5)
    # all-reduce
    _box(ax, 4.8, 0.45, 1.8, 0.8, "all-reduce\n按位置相加", fc="#fee2e2",
         ec="#dc2626", fs=8)
    _arrow(ax, (2.1, 0.75), (4.8, 0.85), color=C_HL, lw=1.5)
    _arrow(ax, (4.2, 0.75), (4.8, 0.85), color=C_GREEN, lw=1.5)
    _box(ax, 7.3, 0.6, 1.35, 0.55, "完整输出\n[1, out]", fc="#fce7f3", fs=8.5)
    _arrow(ax, (6.6, 0.85), (7.3, 0.85), color=C_MAIN, lw=1.5)

    ax.text(0.3, 0.08, "列并行切输出维、行并行切输入维，两者对偶，"
            "通信原语恰好互为相反操作",
            fontsize=9, color="#334155")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "linear_shard.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT / "linear_shard.png")


def draw_mlp_linear_graph() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 4.6), dpi=160)
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    _box(ax, 0.15, 1.95, 1.05, 0.7, "输入 x", "#e0f2fe", C_MAIN, fs=8.6)
    _box(ax, 1.45, 1.95, 1.3, 0.7, "input_layernorm\nRMSNorm", "#f1f5f9",
         C_DIM, fs=8.2)
    _box(ax, 3.0, 1.95, 1.6, 0.7, "qkv_proj\n列并行", "#fef3c7", C_HL, fs=8.2)
    _box(ax, 4.85, 1.95, 1.25, 0.7, "attention\nq/k/v", "#f1f5f9", C_DIM,
         fs=8.2)
    _box(ax, 6.35, 1.95, 1.4, 0.7, "o_proj\n行并行", "#ecfdf5", C_GREEN,
         fs=8.2)
    _box(ax, 8.0, 1.95, 1.3, 0.7, "post_layernorm\nRMSNorm", "#f1f5f9",
         C_DIM, fs=8.2)
    _box(ax, 9.55, 1.95, 1.6, 0.7, "gate_up_proj\n列并行 融合两阵", "#fef3c7",
         C_HL, fs=8.2)
    _box(ax, 11.6, 1.95, 1.1, 0.7, "down_proj\n行并行", "#ecfdf5", C_GREEN,
         fs=8.2)
    _box(ax, 10.0, 0.75, 1.5, 0.55, "silu_and_mul\n一个融合内核", "#fce7f3",
         "#dc2626", fs=8.2)
    _box(ax, 11.8, 0.75, 0.8, 0.4, "残差+", "#f1f5f9", C_DIM, fs=8)

    # 主链箭头
    _arrow(ax, (1.2, 2.3), (1.45, 2.3), color=C_DIM, lw=1.3)
    _arrow(ax, (2.75, 2.3), (3.0, 2.3), color=C_MAIN, lw=1.3)
    _arrow(ax, (4.6, 2.3), (4.85, 2.3), color=C_MAIN, lw=1.3)
    _arrow(ax, (6.1, 2.3), (6.35, 2.3), color=C_MAIN, lw=1.3)
    _arrow(ax, (7.75, 2.3), (8.0, 2.3), color=C_MAIN, lw=1.3)
    _arrow(ax, (9.3, 2.3), (9.55, 2.3), color=C_MAIN, lw=1.3)
    _arrow(ax, (11.15, 2.3), (11.6, 2.3), color=C_MAIN, lw=1.3)
    # gate_up → silu_and_mul
    _arrow(ax, (10.35, 1.95), (10.4, 1.3), color="#dc2626", lw=1.4)
    # silu_and_mul → down_proj
    _arrow(ax, (11.5, 1.05), (12.1, 1.95), color="#dc2626", lw=1.4, curved=0.15)
    # down_proj → 残差
    _arrow(ax, (12.15, 2.3), (11.85, 2.3), color=C_MAIN, lw=1.3, curved=-0.2)

    ax.text(0.15, 3.7, "qkv 输出不拼回，直接在本地拆 q/k/v 进 attention",
            fontsize=8.8, color="#334155")
    ax.text(0.15, 3.35, "o_proj 输出先 all-reduce 再参与残差",
            fontsize=8.8, color=C_GREEN)
    ax.text(0.15, 3.0, "gate_up 的两段输出直接进 silu_and_mul，"
            "中间张量不再拆分", fontsize=8.8, color="#dc2626")
    ax.text(5.0, 0.3, "残差相加发生在每两个归一化之间",
            fontsize=8.8, color="#64748b")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "mlp_linear_graph.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("saved", OUT / "mlp_linear_graph.png")


if __name__ == "__main__":
    draw_linear_shard()
    draw_mlp_linear_graph()
