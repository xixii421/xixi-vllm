"""生成词表文档的两张结构示意图：嵌入查表、LM Head。

用法（在 vllm-v4 下）：
    python docs/posts/gen_vocab_figs.py

输出到 docs/posts/：
    embedding_lookup.png
    lm_head.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent

_FONT = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
fm.fontManager.addfont(_FONT)
plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False

C_MAIN = "#2563eb"
C_HL = "#f59e0b"
C_DIM = "#9ca3af"
C_BG = "#f8fafc"


def _box(ax, x, y, w, h, text, fc=C_BG, ec=C_MAIN, fs=10, tc="black",
         lw=1.4, rounded=0.05):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounded}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3)


def _arrow(ax, p0, p1, color=C_MAIN, lw=1.6, style="-|>", shrink=1.0):
    ar = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, zorder=4, shrinkA=shrink, shrinkB=shrink,
    )
    ax.add_patch(ar)


def draw_embedding_lookup() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.6), dpi=160)
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    _box(ax, 0.2, 3.9, 2.0, 1.3, "输入 token\n[1, 2, 3]", fc="#e0f2fe")
    tokens = ["0  [BOS]", "1  我", "2  爱", "3  中国", "4  [EOS]"]
    for i, t in enumerate(tokens):
        y = 3.55 - i * 0.85
        _box(ax, 0.35, y, 1.7, 0.6, t, fc="#f1f5f9", ec=C_DIM, fs=9)

    mat_x, cell_w, cell_h = 3.0, 0.62, 0.62
    row_head = ["第 0 行", "第 1 行", "第 2 行", "第 3 行", "第 4 行"]
    vals = [
        ["0.1", "0.2", "0.3", "0.4"],
        ["0.5", "-0.1", "0.8", "0.2"],
        ["-0.3", "0.9", "0.1", "-0.5"],
        ["0.7", "0.0", "-0.2", "0.6"],
        ["0.2", "0.8", "-0.4", "0.1"],
    ]
    hl_rows = {1, 2, 3}
    for r in range(5):
        for c in range(5):
            x = mat_x + c * cell_w
            y = 4.15 - r * cell_h
            if c == 0:
                fc = "#fef3c7" if r in hl_rows else "#f8fafc"
                ec = C_HL if r in hl_rows else C_DIM
                _box(ax, x, y, cell_w, cell_h, row_head[r], fc=fc, ec=ec, fs=8.5)
            else:
                fc = "#fef3c7" if r in hl_rows else "#f1f5f9"
                ec = C_HL if r in hl_rows else "#cbd5e1"
                _box(ax, x, y, cell_w, cell_h, vals[r][c - 1], fc=fc,
                     ec=ec, fs=9)
    _box(ax, mat_x, 5.0, 5 * cell_w, 0.5,
         "词嵌入表 [vocab=5, hidden=4]", fc="#dbeafe", ec=C_MAIN, fs=10)

    for i, ti in enumerate([1, 2, 3]):
        ty = 3.55 - ti * 0.85 + 0.3
        my = 4.15 - ti * cell_h + cell_h / 2
        _arrow(ax, (2.05, ty), (mat_x + 0.31, my), color=C_HL, lw=1.8)

    out_x, out_y = 8.5, 1.0
    _box(ax, out_x, 4.4, 2.4, 0.6, "查出的嵌入向量\n[3, 4]", fc="#dbeafe")
    out_vals = [
        ["0.5", "-0.1", "0.8", "0.2"],
        ["-0.3", "0.9", "0.1", "-0.5"],
        ["0.7", "0.0", "-0.2", "0.6"],
    ]
    for r in range(3):
        for c in range(4):
            x = out_x + c * 0.6
            y = out_y + (2 - r) * 0.62
            _box(ax, x, y, 0.6, 0.62, out_vals[r][c], fc="#fef3c7",
                 ec=C_HL, fs=9)
    _arrow(ax, (6.1, 2.4), (out_x - 0.1, 2.9), color=C_MAIN, lw=1.8)

    ax.text(0.2, 0.12, "查表就是从词表矩阵中取出 token 对应的那一行，"
            "表是训练结束后固定不变的。",
            fontsize=9.5, color="#334155")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "embedding_lookup.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("saved", OUT / "embedding_lookup.png")


def draw_lm_head() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.4), dpi=160)
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    _box(ax, 0.2, 3.9, 1.9, 1.1, "最后一个 token\n的隐藏向量 h\n[hidden]", fc="#e0f2fe")
    for i in range(8):
        y = 3.35 - i * 0.28
        _box(ax, 0.45, y, 1.35, 0.24, "", fc="#dbeafe", ec="#93c5fd", lw=0.8)
    ax.text(0.45, 0.75, "h 的 8 个分量示意", fontsize=8.5, color="#64748b")

    mat_x = 3.0
    cell_w, cell_h, rows, cols = 0.56, 0.5, 8, 5
    for r in range(rows):
        for c in range(cols):
            x = mat_x + c * cell_w
            y = 4.55 - r * cell_h
            _box(ax, x, y, cell_w, cell_h, "", fc="#f1f5f9",
                 ec="#cbd5e1", lw=0.7)
    _box(ax, mat_x, 5.0, cols * cell_w, 0.45,
         "LM Head 权重 [V, hidden]", fc="#dbeafe", ec=C_MAIN, fs=9.5)
    ax.text(mat_x, 4.75 - rows * cell_h, "每一行是一个词的方向",
            fontsize=8.5, color="#64748b")

    _arrow(ax, (2.15, 2.6), (mat_x - 0.1, 3.4), color=C_MAIN, lw=1.8)

    logits = [2.1, 0.5, 3.2, 1.1, -0.5, 0.8, -1.2, 0.3]
    labels = ["[BOS]", "我", "爱", "中国", "[EOS]", "你", "好", "的"]
    bar_x, bar_w = 7.1, 2.2
    _box(ax, bar_x, 4.55, bar_w, 0.45, "logits 分数（每个词一个）",
         fc="#dbeafe", ec=C_MAIN, fs=9.5)
    mx = max(logits)
    for i, (v, lb) in enumerate(zip(logits, labels)):
        y = 4.25 - i * 0.48
        w = max(v / mx * 1.9, 0.06)
        color = C_HL if v == mx else "#bfdbfe"
        ax.add_patch(Rectangle((bar_x, y), w, 0.32, fc=color, ec="none", zorder=2))
        ax.text(bar_x + w + 0.05, y + 0.09, f"{lb}  {v}", fontsize=8.5,
                va="center", color="#334155", zorder=3)
    ax.text(bar_x, 0.28, "分数最高的词 = 预测的下一个词", fontsize=9,
            color=C_MAIN)

    ax.text(0.2, 0.1, "LM Head 就是一次矩阵乘，把隐藏向量映射成覆盖整个"
            "词典的分数。", fontsize=9.5, color="#334155")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "lm_head.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT / "lm_head.png")


if __name__ == "__main__":
    draw_embedding_lookup()
    draw_lm_head()
