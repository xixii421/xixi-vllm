"""
用法:  python -m vllm-v0.profile.micro --model ~/huggingface/Qwen3-0.6B
"""

from __future__ import annotations

import argparse, math, sys, time
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent  # project root
sys.path.append(str(_PROJ / "vllm-v0"))  # for src.xxx
sys.path.append(str(_PROJ))              # for profiler

import torch
from src.config import EngineConfig
from src.engine import Engine


def run(model: str, ctx: int = 8192, warmup: int = 5, rep: int = 20,
        save: str = "") -> None:
    eng = Engine(EngineConfig(model=model, enforce_eager=True, context_len=ctx))
    cfg = eng.config.hf_config
    D, I, NH, NL = cfg.hidden_size, cfg.intermediate_size, \
                   cfg.num_attention_heads, cfg.num_hidden_layers
    HD = getattr(cfg, "head_dim", D // NH)

    # 理论 FLOP: qkv+out+mlp+norm
    gemm_tok = D * (4 * NH * HD) + 6 * D * I + 4 * D  # GEMM flop/token
    sdpa_tok = HD * NH  # SDPA flop/token‑pair

    all_ids = torch.randint(0, 32000, (ctx,), device=eng.device, dtype=torch.long)
    all_pos = torch.arange(ctx, device=eng.device)
    for _ in range(warmup):
        eng.model(all_ids[:ctx].unsqueeze(0), all_pos[:ctx].unsqueeze(0))
    torch.cuda.synchronize()

    S_list = [128, 256, 512, 1024, 2048, 4096, 8192]
    S_list = [s for s in S_list if s <= ctx]
    S_vals, T_vals = [], []

    print(f"\n{'S':>6s}  {'fwd(ms)':>8s}  {'ratio':>6s}  {'SD/GM%':>7s}")
    print("-" * 33)
    for S in S_list:
        ids = all_ids[:S].unsqueeze(0); pos = all_pos[:S].unsqueeze(0)
        ts = []
        for _ in range(rep):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            eng.model(ids, pos); torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        med = sorted(ts)[rep // 2] * 1000
        ratio = med / T_vals[0] if T_vals else 1.0
        sd_frac = S * sdpa_tok / gemm_tok * 100
        print(f"{S:>6d}  {med:>8.2f}  {ratio:>5.2f}x  {sd_frac:>6.1f}%")
        S_vals.append(S); T_vals.append(med)

    cross = int(gemm_tok / sdpa_tok)
    print(f"\n── 理论 ── GEMM={gemm_tok/1e6:.1f}M/tok  "
          f"SDPA={sdpa_tok/1e3:.1f}K/pair  临界S={cross:,}")

    # ── 可视化：左=延迟曲线，右=归一化 t/S ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1.plot(S_vals, T_vals, "bo-", ms=6, label="measured")
        ax1.set(xlabel="Sequence length S", ylabel="Latency (ms)",
                title="model.forward latency")
        ax1.grid(True, alpha=0.3); ax1.legend()
        t_per_s = [t / s for t, s in zip(T_vals, S_vals)]
        ax2.plot(S_vals, t_per_s, "ro-", ms=6, label="t/S (ms/tok)")
        ax2.axhline(y=t_per_s[0], color="gray", ls="--", label="O(S) baseline")
        ax2.set(xlabel="Sequence length S", ylabel="ms / token",
                title="Normalized latency (t/S)")
        ax2.grid(True, alpha=0.3); ax2.legend()
        fig.tight_layout()
        out = save or str(Path(__file__).resolve().parent / "micro_v0.png")
        fig.savefig(out, dpi=120)
        print(f"\n[图] {out}")
        plt.close(fig)
    except ImportError:
        print("\n[!] matplotlib 未安装，跳过图表输出")


def main():
    p = argparse.ArgumentParser(description="v0 micro 基准")
    p.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--save", default="", help="图表保存路径")
    args = p.parse_args()
    run(args.model, args.ctx, save=args.save)


if __name__ == "__main__":
    main()
