"""生成 sampler 文档里的示意图。

用法（在 vllm-v0 下）：
    .venv/bin/python docs/posts/gen_sampler_figs.py

输出到 docs/posts/：
    sampler_lineage.png      采样策略发展脉络
    temperature_effect.png   temperature 对 softmax 分布的影响
    topk_vs_topp.png         top-k 固定个数 vs top-p 动态个数
    sampler_pipeline.png     v0 Sampler 完整链路与贪心快路径
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent

_FONT = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
fm.fontManager.addfont(_FONT)
plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False

C_MAIN = "#2563eb"
C_HL = "#f59e0b"
C_DIM = "#9ca3af"
C_BG = "#f8fafc"
C_OK = "#059669"
C_RED = "#dc2626"
C_PURPLE = "#7c3aed"


def _box(ax, x, y, w, h, text, fc=C_BG, ec=C_MAIN, fs=10, tc="black", lw=1.4, rounded=0.04):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={rounded}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(box)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3)


def _arrow(ax, p0, p1, color=C_MAIN, lw=1.6, style="-|>"):
    ar = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=style,
        mutation_scale=13,
        linewidth=lw,
        color=color,
        zorder=4,
        shrinkA=1.0,
        shrinkB=1.0,
    )
    ax.add_patch(ar)


def draw_lineage() -> None:
    fig, ax = plt.subplots(figsize=(11.8, 4.6), dpi=160)
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    items = [
        (0.15, "贪心 / beam\n退化、重复"),
        (2.35, "temperature\nHinton 2015"),
        (4.55, "top-k\nFan ACL 2018"),
        (6.75, "top-p nucleus\nHoltzman ICLR 2020"),
        (8.95, "v0 Sampler\n三件套合一"),
    ]
    for i, (x, t) in enumerate(items):
        _box(ax, x, 1.7, 2.1, 1.7, t, fc="#e0f2fe" if i < 4 else "#fef3c7",
             ec=C_HL if i == 4 else C_MAIN, fs=8.5)
        if i < len(items) - 1:
            _arrow(ax, (x + 2.1, 2.55), (items[i + 1][0], 2.55))

    ax.text(
        5.9, 0.95,
        "贪心退化成重复短语，随机采样又被分布尾部拖累。",
        ha="center", fontsize=9.5, color="#334155",
    )
    ax.text(
        5.9, 0.55,
        "temperature 先缩放分布，top-k 固定截断，top-p 按累积概率动态截断。",
        ha="center", fontsize=9.5, color="#334155",
    )
    ax.text(
        5.9, 0.15,
        "v0 把 temperature、top-k、top-p 统一放进一个 forward。",
        ha="center", fontsize=9.5, color="#64748b",
    )
    fig.savefig(OUT / "sampler_lineage.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT / "sampler_lineage.png")


def draw_temperature_effect() -> None:
    import numpy as np

    logits = np.array([2.4, 1.2, 0.6, 0.0, -0.6, -1.2])

    def softmax(z, t):
        z = np.array(z) / t
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.0), dpi=160)
    for ax, (t, title) in zip(
        axes,
        [
            (0.5, "T = 0.5\n高分被放大，分布尖锐"),
            (1.0, "T = 1.0\n原分布"),
            (2.0, "T = 2.0\n差异被压平，分布平坦"),
        ],
    ):
        probs = softmax(logits, t)
        bars = ax.bar(range(6), probs, color=["#2563eb" if i == 0 else "#c7d7f5" for i in range(6)])
        ax.set_xticks(range(6))
        ax.set_xticklabels([f"tok{i}" for i in range(6)], fontsize=8)
        ax.set_ylim(0, 0.85)
        ax.set_title(title, fontsize=9.5)
        ax.tick_params(labelsize=7)
        for b, p in zip(bars, probs):
            ax.text(b.get_x() + b.get_width() / 2, p + 0.015, f"{p:.2f}", ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT / "temperature_effect.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT / "temperature_effect.png")


def draw_topk_vs_topp() -> None:
    import numpy as np

    # 左：平坦分布，右：尖锐分布
    flat = np.array([0.13, 0.12, 0.11, 0.10, 0.10, 0.09, 0.09, 0.08, 0.07, 0.06, 0.03, 0.02])
    sharp = np.array([0.62, 0.28, 0.04, 0.02, 0.01, 0.01, 0.01, 0.005, 0.003, 0.001, 0.001, 0.000])

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), dpi=160)

    def paint(ax, probs, keep, title):
        colors = ["#2563eb" if i < keep else "#d6dbe3" for i in range(len(probs))]
        ax.bar(range(len(probs)), probs, color=colors)
        ax.set_xticks(range(len(probs)))
        ax.set_xticklabels([str(i) for i in range(len(probs))], fontsize=7)
        ax.set_ylim(0, 0.7)
        ax.set_title(title, fontsize=9.5)
        ax.tick_params(labelsize=7)

    # 平坦分布：top-k=4 只保留 4 个，top-p=0.9 要保留 8 个
    paint(axes[0], flat, 4, "分布平坦：top-k=4 砍掉太多\ntop-p=0.9 动态保留约 8 个")
    axes[0].axvline(3.5, color=C_HL, lw=1.4, ls="--")
    axes[0].text(3.5, 0.62, "k=4 截断", color=C_HL, fontsize=8, ha="center")

    # 尖锐分布：top-k=4 多保留 2 个低分，top-p=0.9 只保留 2 个
    paint(axes[1], sharp, 4, "分布尖锐：top-k=4 多保留低分\n top-p=0.9 只保留 2 个")
    axes[1].axvline(3.5, color=C_HL, lw=1.4, ls="--")
    axes[1].text(3.5, 0.52, "k=4 截断", color=C_HL, fontsize=8, ha="center")

    fig.tight_layout()
    fig.savefig(OUT / "topk_vs_topp.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT / "topk_vs_topp.png")


def draw_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.8, 4.8), dpi=160)
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    _box(ax, 0.2, 3.85, 2.2, 0.75, "logits\n[B, vocab]", fc="#e0f2fe", fs=9)

    # 判断分支
    _box(ax, 3.0, 3.85, 2.2, 0.75, "温度、top-k、top-p\n全为 None？", fc="#fef3c7", ec=C_HL, fs=9)
    _arrow(ax, (2.4, 4.2), (3.0, 4.2))

    # 快路径：argmax
    _box(ax, 6.0, 4.6, 2.4, 0.75, "argmax\n返回最高分 token", fc="#dcfce7", ec=C_OK, fs=9)
    _arrow(ax, (5.2, 4.2), (6.0, 4.75), color=C_OK)

    # 完整链路：五步
    steps = [
        "① float32",
        "② ÷ temperature",
        "③ top-k / top-p 设 -inf",
        "④ softmax",
        "⑤ multinomial",
    ]
    x0 = 6.0
    for i, s in enumerate(steps):
        y = 3.5 - i * 0.68
        fc = "#dbeafe" if i in (0, 1, 3, 4) else "#fde68a"
        ec = C_MAIN if i in (0, 1, 3, 4) else C_HL
        _box(ax, x0, y, 2.4, 0.58, s, fc=fc, ec=ec, fs=9)
        if i < len(steps) - 1:
            _arrow(ax, (x0 + 1.2, y), (x0 + 1.2, y - 0.1))
    _arrow(ax, (5.2, 3.85), (x0, 3.85))

    _box(ax, x0, 0.15, 2.4, 0.7, "next_token_id", fc="#e0f2fe", fs=9)
    _arrow(ax, (x0 + 1.2, 3.5 - 4 * 0.68), (x0 + 1.2, 0.85))

    ax.text(
        5.9, 0.35,
        "一条快路径只做 argmax，一条完整链路走五步。",
        ha="center", fontsize=9.5, color="#334155",
    )
    fig.savefig(OUT / "sampler_pipeline.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT / "sampler_pipeline.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    draw_lineage()
    draw_temperature_effect()
    draw_topk_vs_topp()
    draw_pipeline()
