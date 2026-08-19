"""生成 CG vs eager 对比图（decode 吞吐 与 单步延迟）。"""

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

C_MAIN = "#2563eb"
C_HL = "#f59e0b"
C_GREEN = "#16a34a"
C_RED = "#dc2626"

# (tp, cg) -> decode tok/s 按 n=1,4,8,16
DATA = {
    ("tp1", "eager"): (63.8, 240.3, 474.3, 778.6),
    ("tp1", "cg"): (437.9, 1471.1, 2813.6, 4398.2),
    ("tp2", "eager"): (40.0, 149.8, 305.5, 603.1),
    ("tp2", "cg"): (144.9, 546.1, 1069.2, 1944.9),
}
STEPS = {  # n=1 单步延迟 ms
    ("tp1", "eager"): 15.42,
    ("tp1", "cg"): 2.27,
    ("tp2", "eager"): 24.77,
    ("tp2", "cg"): 6.90,
}
N = [1, 4, 8, 16]


def draw_throughput() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=160)
    labels = [("tp1", "eager"), ("tp1", "cg"), ("tp2", "eager"), ("tp2", "cg")]
    colors = ["#93c5fd", C_MAIN, "#fda4af", C_RED]
    width = 0.2
    x = np.arange(len(N))
    for i, (key, color) in enumerate(zip(labels, colors)):
        vals = DATA[key]
        ax.bar(x + (i - 1.5) * width, vals, width,
               label=f"{key[0]} {key[1]}", color=color, zorder=2)
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 1.5) * width, v + 30, f"{v:.0f}",
                    ha="center", fontsize=7.5, color="#334155")
    ax.set_xticks(x)
    ax.set_xticklabels([f"并发 {n} 请求" for n in N])
    ax.set_ylabel("decode 吞吐 (tok/s)")
    ax.set_title("Qwen3-0.6B  TP=1/2  CG vs eager  decode 吞吐")
    ax.legend(fontsize=9)
    ax.grid(axis="y", color="#e2e8f0", zorder=0)
    ax.set_ylim(0, 4800)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / "cg_vs_eager_throughput.png", facecolor="white")
    plt.close(fig)
    print("saved", OUT / "cg_vs_eager_throughput.png")


def draw_step_latency() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=160)
    labels = [("tp1", "eager"), ("tp1", "cg"), ("tp2", "eager"), ("tp2", "cg")]
    colors = ["#93c5fd", C_MAIN, "#fda4af", C_RED]
    vals = [STEPS[k] for k in labels]
    bars = ax.bar([f"{k[0]} {k[1]}" for k in labels], vals, 0.6, color=colors, zorder=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.2f}",
                ha="center", fontsize=10, color="#334155")
    ax.set_ylabel("单步 decode 墙钟 (ms)")
    ax.set_title("Qwen3-0.6B  单请求 decode 单步延迟（中位数）")
    ax.grid(axis="y", color="#e2e8f0", zorder=0)
    ax.set_ylim(0, 28)
    ax.text(0.01, 26.5,
            "tp1 eager→cg 快 6.8x，tp2 eager→cg 快 3.6x\n"
            "CG 消除的是每步 CPU launch 开销，剩余是 GPU 实际执行时间",
            fontsize=9, color="#334155")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / "cg_vs_eager_step.png", facecolor="white")
    plt.close(fig)
    print("saved", OUT / "cg_vs_eager_step.png")


if __name__ == "__main__":
    draw_throughput()
    draw_step_latency()
