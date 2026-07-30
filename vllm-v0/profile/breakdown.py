"""v0 瓶颈拆解 
用法:  python -m vllm-v0.profile.breakdown --model ~/huggingface/Qwen3-0.6B
"""

from __future__ import annotations

import argparse, sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(_PROJ / "vllm-v0"))
sys.path.append(str(_PROJ))

import torch
from src.config import EngineConfig
from src.engine import Engine

E = torch.cuda.Event


def _t(fn, *a, **kw):
    s, e = E(enable_timing=True), E(enable_timing=True)
    s.record(); out = fn(*a, **kw); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e), out


def run(model: str, ctx: int = 8192, warmup: int = 5, rep: int = 20) -> None:
    eng = Engine(EngineConfig(model=model, enforce_eager=True, context_len=ctx))
    lay = eng.model.model.layers[0]
    all_ids = torch.randint(0, 32000, (ctx,), device=eng.device, dtype=torch.long)
    all_pos = torch.arange(ctx, device=eng.device)
    emb = eng.model.model.embed_tokens

    S_list = [s for s in [128, 256, 512, 1024, 2048, 4096, 8192] if s <= ctx]

    for _ in range(warmup):
        lay(all_pos[:max(S_list)].unsqueeze(0),
            emb(all_ids[:max(S_list)].unsqueeze(0)), None)
    torch.cuda.synchronize()

    print(f"\n{'S':>6s}  {'attn(ms)':>9s}  {'mlp(ms)':>9s}  {'norm(ms)':>9s}  "
          f"{'total':>8s}  {'attn%':>7s}")
    print("-" * 56)

    for S in S_list:
        h = emb(all_ids[:S].unsqueeze(0)); pos = all_pos[:S].unsqueeze(0)
        a_sum = m_sum = n_sum = t_sum = 0.0
        for _ in range(rep):
            # total time (full forward)
            tt, _ = _t(lay, pos, h, None)
            t_sum += tt

            # 分项: pre‑norm → attn → post‑norm → mlp
            # 首层: input_layernorm 无 residual → 返回 tensor
            n1_t, h_n = _t(lay.input_layernorm, h)  # h 作为 residual
            a_t, attn_out = _t(lay.self_attn, pos, h_n)
            # post_attn_norm 带 residual → 返回 (out, residual) 二元组
            n2_t, (h2, _) = _t(lay.post_attention_layernorm, attn_out, h)
            m_t, _ = _t(lay.mlp, h2)
            a_sum += a_t; m_sum += m_t; n_sum += n1_t + n2_t

        a_ms = a_sum / rep; m_ms = m_sum / rep
        n_ms = n_sum / rep; t_ms = t_sum / rep
        print(f"{S:>6d}  {a_ms:>9.2f}  {m_ms:>9.2f}  {n_ms:>9.2f}  "
              f"{t_ms:>7.2f}ms {a_ms/t_ms*100:>6.1f}%")

    print(f"\n── 诊断 ──")
    print(f"  v0 无 KV cache: 每步全量重算 Q*K^T → attention O(S²)")
    print(f"  S↑ → attn 占比↑ → SDPA 是瓶颈. GEMM(S) 的增长远慢于 SDPA(S²)")
    print(f"  v1 KV cache: 缓存在步间复用 K/V，解码步仅计算 1 个新 token")
    print(f"  → 预期 v1 解码延迟基本恒定，不随历史长度 S 增长")


def main():
    p = argparse.ArgumentParser(description="v0 瓶颈拆解")
    p.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    p.add_argument("--ctx", type=int, default=8192)
    args = p.parse_args()
    run(args.model, args.ctx)


if __name__ == "__main__":
    main()
