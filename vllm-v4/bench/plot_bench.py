"""IEEE 风格性能图表生成器。

读取 bench/out/bench.jsonl（bench_parallel / bench_dp / bench_vllm_official 结构化输出），
生成 6 张英文图表（SVG + PNG）到 docs/figures/。

风格：serif 字体、隐藏上/右 spine、细灰网格、色盲友好配色 + 灰阶填充，贴合 IEEE 期刊图例。

用法（在 vllm-v4/ 下）:
    ../.venv2/bin/python bench/plot_bench.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "out"
JSONL = OUT / "bench.jsonl"
FIGDIR = ROOT / "docs" / "figures"

# Okabe-Ito 色盲友好配色 + 对应灰阶填充。
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
HATCHES = ["", "///", "xx", "..", "OO", "\\\\"]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

MODE_LABELS = {
    "eager": "Eager",
    "cg": "CUDA Graph",
    "compile": "Compile",
    "cg+compile": "CG + Compile",
}


def load_records() -> dict:
    """返回 {kind: dict}；kind ∈ {par, dp}。

    par[(engine, model, tp, pp, mode)] = {ttft_ms, decode_tps, batch_tps}
    dp[(model, dp, mode)] = {throughput_tps}
    """
    par: dict = {}
    dp: dict = {}
    if not JSONL.exists():
        print(f"WARN: {JSONL} not found")
        return {"par": par, "dp": dp}
    for line in JSONL.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "throughput_tps" in r:
            dp[(r["model"], r["dp"], r["mode"])] = {"throughput_tps": r["throughput_tps"]}
        else:
            key = (r["engine"], r["model"], r["tp"], r["pp"], r["mode"])
            par[key] = {
                "ttft_ms": r.get("ttft_ms", 0.0),
                "decode_tps": r.get("decode_tps", 0.0),
                "batch_tps": r.get("batch_tps", 0.0),
            }
    return {"par": par, "dp": dp}


def style_axes(ax, ylabel: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, color="#cccccc")
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel)


def grouped_bar(ax, categories, series, values, *, bar_width=0.34):
    """分组柱状图。categories 为 x 轴，series 为图例，values[(cat, ser)]=float。"""
    x = np.arange(len(categories))
    n = len(series)
    w = bar_width / n
    for i, s in enumerate(series):
        vals = [values.get((c, s), 0.0) for c in categories]
        ax.bar(
            x + (i - (n - 1) / 2) * w,
            vals,
            w,
            label=s,
            color=COLORS[i % len(COLORS)],
            hatch=HATCHES[i % len(HATCHES)],
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.margins(x=0.05)


def save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  → docs/figures/{name}.svg/.png")


def get_par(par, engine, model, tp, pp, mode, metric):
    rec = par.get((engine, model, tp, pp, mode))
    return rec[metric] if rec else 0.0


# --------------------------------------------------------------------------- #
# 图 1：CG 对 decode 的收益（8B）
# --------------------------------------------------------------------------- #
def fig_cg_speedup(par) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    cats = ["Single", "TP=2", "PP=2"]
    layouts = [(1, 1), (2, 1), (1, 2)]
    series = ["Eager", "CUDA Graph"]
    values = {}
    for cat, (tp, pp) in zip(cats, layouts):
        values[(cat, "Eager")] = get_par(par, "v4", "Qwen3-8B", tp, pp, "eager", "decode_tps")
        values[(cat, "CUDA Graph")] = get_par(par, "v4", "Qwen3-8B", tp, pp, "cg", "decode_tps")
    grouped_bar(ax, cats, series, values)
    style_axes(ax, "Decode throughput (tok/s)")
    ax.legend(frameon=False)
    ax.set_title("CUDA Graph Speedup by Parallelism (Qwen3-8B)")
    save(fig, "fig1_cg_speedup")


# --------------------------------------------------------------------------- #
# 图 2：compile 对 TTFT 的收益（8B）
# --------------------------------------------------------------------------- #
def fig_compile_ttft(par) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    # PP=2 的 eager+compile 超时（见文档），此处只画单卡与 TP=2。
    cats = ["Single", "TP=2"]
    layouts = [(1, 1), (2, 1)]
    series = ["Eager", "Compile"]
    values = {}
    for cat, (tp, pp) in zip(cats, layouts):
        values[(cat, "Eager")] = get_par(par, "v4", "Qwen3-8B", tp, pp, "eager", "ttft_ms")
        values[(cat, "Compile")] = get_par(par, "v4", "Qwen3-8B", tp, pp, "compile", "ttft_ms")
    grouped_bar(ax, cats, series, values)
    style_axes(ax, "TTFT (ms)")
    ax.legend(frameon=False)
    ax.set_title("Prefill TTFT: Eager vs Compile (Qwen3-8B)")
    save(fig, "fig2_compile_ttft")


# --------------------------------------------------------------------------- #
# 图 3：CG 模式下相对单卡的扩展效率（8B）
# --------------------------------------------------------------------------- #
def fig_scaling(par) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    cats = ["Single", "TP=2", "PP=2"]
    layouts = [(1, 1), (2, 1), (1, 2)]
    base = get_par(par, "v4", "Qwen3-8B", 1, 1, "cg", "batch_tps")
    vals = []
    for tp, pp in layouts:
        t = get_par(par, "v4", "Qwen3-8B", tp, pp, "cg", "batch_tps")
        vals.append(t / base if base else 0.0)
    x = np.arange(len(cats))
    ax.bar(x, vals, 0.4, color=COLORS[0], hatch=HATCHES[1], edgecolor="black", linewidth=0.4)
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.03, f"{v:.2f}x", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    style_axes(ax, "Relative batch throughput")
    ax.set_title("Scaling Efficiency vs Single GPU (Qwen3-8B, CUDA Graph)")
    save(fig, "fig3_scaling")


# --------------------------------------------------------------------------- #
# 图 4：v4 vs vLLM decode 吞吐（8B）
# --------------------------------------------------------------------------- #
def fig_vs_vllm(par) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    cats = ["Single\nEager", "Single\nCUDA Graph", "TP=2\nEager", "TP=2\nCUDA Graph"]
    cfgs = [(1, "eager"), (1, "cg"), (2, "eager"), (2, "cg")]
    series = ["v4", "vLLM"]
    values = {}
    for cat, (tp, mode) in zip(cats, cfgs):
        values[(cat, "v4")] = get_par(par, "v4", "Qwen3-8B", tp, 1, mode, "decode_tps")
        values[(cat, "vLLM")] = get_par(par, "vllm", "Qwen3-8B", tp, 1, mode, "decode_tps")
    grouped_bar(ax, cats, series, values)
    style_axes(ax, "Decode throughput (tok/s)")
    ax.legend(frameon=False)
    ax.set_title("v4 vs vLLM: Decode Throughput (Qwen3-8B)")
    save(fig, "fig4_vs_vllm")


# --------------------------------------------------------------------------- #
# 图 5：DP 吞吐（4B）
# --------------------------------------------------------------------------- #
def fig_dp(dp) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    cats = ["DP=1", "DP=2"]
    series = ["Eager", "Compile"]
    values = {}
    for cat, d in zip(cats, [1, 2]):
        values[(cat, "Eager")] = dp.get(("Qwen3-4B", d, "eager"), {}).get("throughput_tps", 0.0)
        values[(cat, "Compile")] = dp.get(("Qwen3-4B", d, "compile"), {}).get("throughput_tps", 0.0)
    grouped_bar(ax, cats, series, values)
    cg1 = dp.get(("Qwen3-4B", 1, "cg"), {}).get("throughput_tps", 0.0)
    ax.annotate(
        f"CUDA Graph (DP=1): {cg1:.0f} tok/s\n(DP=2 CG: OOM on 12 GB GPU)",
        xy=(0.98, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#999999", lw=0.5),
    )
    style_axes(ax, "Aggregate throughput (tok/s)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Data Parallelism Throughput (Qwen3-4B)")
    save(fig, "fig5_dp")


# --------------------------------------------------------------------------- #
# 图 6：14B PP=2（decode + TTFT 双面板）
# --------------------------------------------------------------------------- #
def fig_pp14b(par) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.4))
    modes = ["eager", "cg", "compile", "cg+compile"]
    labels = [MODE_LABELS[m] for m in modes]
    decode = [get_par(par, "v4", "Qwen3-14B", 1, 2, m, "decode_tps") for m in modes]
    ttft = [get_par(par, "v4", "Qwen3-14B", 1, 2, m, "ttft_ms") for m in modes]

    x = np.arange(len(modes))
    ax = axes[0]
    ax.bar(x, decode, 0.5, color=COLORS[0], hatch=HATCHES[1], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    style_axes(ax, "Decode (tok/s)")
    ax.set_title("(a) Decode throughput")

    ax = axes[1]
    ax.bar(x, ttft, 0.5, color=COLORS[2], hatch=HATCHES[2], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    style_axes(ax, "TTFT (ms)")
    ax.set_title("(b) Time-to-first-token")

    fig.suptitle("Pipeline Parallelism (Qwen3-14B, PP=2)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig6_pp14b")


def main() -> None:
    data = load_records()
    par, dp = data["par"], data["dp"]
    if not par and not dp:
        print("No benchmark records found; run bench/run_matrix_large.sh first.")
        return
    fig_cg_speedup(par)
    fig_compile_ttft(par)
    fig_scaling(par)
    fig_vs_vllm(par)
    fig_dp(dp)
    fig_pp14b(par)
    print("Done.")


if __name__ == "__main__":
    main()
