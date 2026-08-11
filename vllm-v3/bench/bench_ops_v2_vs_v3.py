"""v2 vs v3 算子级性能剖析。

对 v3 优化的每个核心算子做 v2/v3 双实现对比，覆盖 prefill 与 decode 两种形状。
使用与 Qwen3-0.6B 一致的真实维度（hidden=1024, intermediate=3072,
heads=16, kv_heads=8, head_dim=128）。

计时方式：CUDA event，预热后取多次运行的中位数。

模块对照（对应 docs/01-v2性能瓶颈.md 的 A~E）：
  rmsnorm / rope / silu / mlp      → 瓶颈 D（算子碎片）
  kvwrite                          → 瓶颈 C（index_copy_ 落盘）
  attn_prefill / attn_decode       → 瓶颈 A（attention 逐条/SDPA）
  sampler                          → 瓶颈 E（sort + multinomial 同步）
  layer_decode / layer_prefill     → 整层前向（eager 组合路径）

用法:
    python bench_ops_v2_vs_v3.py              # 全部模块
    python bench_ops_v2_vs_v3.py --only rmsnorm,mlp
    python bench_ops_v2_vs_v3.py --iters 200 --warm 50
    python bench_ops_v2_vs_v3.py --device cuda:1
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]  # vllm-v3/


# ---------------------------------------------------------------------------
# 按文件加载 v2/v3 同名 src 模块，避免包名冲突
# ---------------------------------------------------------------------------

_MODULE_CACHE: dict[str, object] = {}


def load_ver(ver: str, rel: str) -> object:
    """从 vllm-{ver}/{rel}.py 加载模块（文件级 import，两版可共存）。"""
    key = f"{ver}:{rel}"
    if key not in _MODULE_CACHE:
        path = ROOT.parent / f"vllm-{ver}" / rel
        spec = importlib.util.spec_from_file_location(key, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULE_CACHE[key] = mod
    return _MODULE_CACHE[key]


def attr(ver: str, rel: str, name: str) -> object:
    return getattr(load_ver(ver, rel), name)


# ---------------------------------------------------------------------------
# 模块工厂：构造相同输入的 v2/v3 实现
# ---------------------------------------------------------------------------

D = dict(device="cuda", dtype=torch.bfloat16)


def make_rmsnorm(ver: str, hidden: int = 1024):
    mod = load_ver(ver, "src/layers/layernorm.py")
    return mod.RMSNorm(hidden, eps=1e-6).to(**D)


def make_rope(ver: str, head_dim: int = 128, max_pos: int = 4096):
    mod = load_ver(ver, "src/layers/rotary.py")
    return mod.RotaryEmbedding(head_dim, head_dim, max_pos, base=1_000_000).to(**D)


def make_mlp(ver: str):
    if ver == "v2":
        mod = load_ver(ver, "src/layers/activation.py")
        act = mod.SiluAndMul()
        gate = torch.nn.Linear(1024, 3072, bias=False).to(**D)
        up = torch.nn.Linear(1024, 3072, bias=False).to(**D)
        down = torch.nn.Linear(3072, 1024, bias=False).to(**D)

        def forward(x):
            return down(act(torch.cat([gate(x), up(x)], dim=-1)))

        return forward

    mod = load_ver(ver, "src/layers/activation.py")
    act = mod.SiluAndMul()
    gu = torch.nn.Linear(1024, 3072 * 2, bias=False).to(**D)
    down = torch.nn.Linear(3072, 1024, bias=False).to(**D)

    def forward(x):
        return down(act(gu(x)))

    return forward


def make_silu(ver: str):
    mod = load_ver(ver, "src/layers/activation.py")
    return mod.SiluAndMul().to(**D)


def make_sampler(ver: str):
    if ver == "v2":
        return load_ver(ver, "src/sampler.py").Sampler()
    return load_ver(ver, "src/engine/sampler.py").Sampler().to(**D)


def make_kvwrite(ver: str, num_kv_heads=8, head_dim=128):
    """构造 KV 落盘实现：v2=index_copy_，v3=Triton store_kvcache。"""
    cache = torch.zeros(4096, num_kv_heads, head_dim, **D)

    if ver == "v2":
        def write(k, v, slots):
            cache.index_copy_(0, slots, k)
            cache.index_copy_(0, slots, v)
    else:
        fn = attr("v3", "src/memory/kv_ops.py", "store_kvcache")

        def write(k, v, slots):
            fn(k, v, cache, cache, slots)

    return write


def make_attn_prefill_v2(num_heads=16, num_kv_heads=8, head_dim=128):
    """v2 prefill：SDPA causal（无前缀缓存时主路径）。"""
    scale = head_dim ** -0.5

    def _gqa(k, v):
        # [T, Hkv, D] → [T, H, D]，模拟 v2 的 _expand_gqa
        groups = num_heads // num_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        return k, v

    def run(q, k, v):
        k, v = _gqa(k, v)
        q = q.unsqueeze(0).transpose(1, 2)  # [1,H,T,D]
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=True
        )
        return out.transpose(1, 2).contiguous()

    return run


def make_attn_decode_v2(num_heads=16, num_kv_heads=8, head_dim=128, block_size=16):
    """v2 decode：逐条 gather + SDPA（v2 的真实主路径）。"""
    scale = head_dim ** -0.5
    pool_k = torch.zeros(8, 1024, block_size, num_kv_heads, head_dim, **D)
    pool_v = torch.zeros_like(pool_k)
    block_table = list(range(256))  # 足够覆盖 ctx=4096（256 块 × 16）

    def gather(layer, table, seq_len):
        table_t = torch.tensor(table[: (seq_len + block_size - 1) // block_size],
                                device="cuda", dtype=torch.long)
        positions = torch.arange(seq_len, device="cuda", dtype=torch.long)
        physical = table_t[positions // block_size]
        offsets = positions % block_size
        return pool_k[layer, physical, offsets], pool_v[layer, physical, offsets]

    def run(q, seq_len):
        outs = []
        for b in range(q.shape[0]):
            k_c, v_c = gather(0, block_table, seq_len)
            # 模拟 v2 的 _expand_gqa：[Hkv,D] → [H,D]
            groups = num_heads // num_kv_heads
            k_dup = k_c.repeat_interleave(groups, dim=1)
            v_dup = v_c.repeat_interleave(groups, dim=1)
            out = torch.nn.functional.scaled_dot_product_attention(
                q[b : b + 1].unsqueeze(0).transpose(1, 2),
                k_dup.unsqueeze(0).transpose(1, 2),
                v_dup.unsqueeze(0).transpose(1, 2),
                scale=scale,
                is_causal=False,
            )
            outs.append(out.transpose(1, 2).contiguous())
        return torch.cat(outs, dim=0)

    return run


def make_attn_prefill_v3(num_heads=16, num_kv_heads=8, head_dim=128, block_size=256):
    """v3 prefill：FlashInfer paged prefill wrapper（续 chunk 主路径）。"""
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    n_blocks = 32
    pool = torch.empty(2, n_blocks, block_size, num_kv_heads, head_dim,
                       dtype=torch.bfloat16, device="cuda")
    k_4d, v_4d = pool[0], pool[1]

    def run(q, seq_len):
        n = q.shape[0]  # 总 token 数（单条=seq_len，多条=sum(seq_len)）
        nb = (seq_len + block_size - 1) // block_size
        wrapper.plan(
            qo_indptr=torch.tensor([0, n], dtype=torch.int32, device="cpu"),
            paged_kv_indices=torch.arange(nb, dtype=torch.int32, device="cuda"),
            paged_kv_indptr=torch.tensor([0, nb], dtype=torch.int32, device="cpu"),
            paged_kv_last_page_len=torch.tensor(
                [(seq_len - 1) % block_size + 1], dtype=torch.int32, device="cpu"
            ),
            num_qo_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim, head_dim_vo=head_dim,
            page_size=block_size, causal=True, sm_scale=head_dim ** -0.5,
            q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
        )
        return wrapper.run(q, (k_4d, v_4d))

    return run


def make_attn_decode_v3(num_heads=16, num_kv_heads=8, head_dim=128, block_size=256):
    """v3 decode：FlashInfer paged decode wrapper（CUDA Graph 内主路径）。"""
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD",
                                                 use_tensor_cores=True)
    n_blocks = 512  # 覆盖 ctx=4096 × bs=32（16 块/seq × 32）
    pool = torch.empty(2, n_blocks, block_size, num_kv_heads, head_dim,
                       dtype=torch.bfloat16, device="cuda")
    k_4d, v_4d = pool[0], pool[1]

    def run(q, seq_len, bs):
        nb = (seq_len + block_size - 1) // block_size
        indptr = torch.tensor([i * nb for i in range(bs + 1)],
                              dtype=torch.int32, device="cpu")
        indices = torch.arange(bs * nb, dtype=torch.int32, device="cuda")
        last = torch.full((bs,), (seq_len - 1) % block_size + 1,
                          dtype=torch.int32, device="cpu")
        wrapper.plan(
            indptr=indptr, indices=indices, last_page_len=last,
            num_qo_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            page_size=block_size, sm_scale=head_dim ** -0.5,
            q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
        )
        return wrapper.run(q, (k_4d, v_4d))

    return run


# ---------------------------------------------------------------------------
# 计时
# ---------------------------------------------------------------------------


def bench(fn, *args, warm: int, iters: int, **kwargs) -> float:
    for _ in range(warm):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def shape_of(*xs) -> str:
    parts = []
    for x in xs:
        parts.append("×".join(str(s) for s in x.shape))
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# 测试项注册
# ---------------------------------------------------------------------------

CASES: dict[str, dict] = {}


def case(name: str, label: str):
    def deco(fn):
        CASES[name] = {"label": label, "fn": fn}
        return fn

    return deco


@case("rmsnorm", "D · RMSNorm：v2 三段式 vs v3 Triton 融合")
def _rmsnorm(args):
    v2, v3 = make_rmsnorm("v2"), make_rmsnorm("v3")
    rows = []
    for shape in [(1, 4096, 1024), (8, 256, 1024), (32, 1, 1024)]:
        x = torch.randn(*shape, **D)
        t2 = bench(v2, x, warm=args.warm, iters=args.iters)
        t3 = bench(v3, x, warm=args.warm, iters=args.iters)
        rows.append((shape_of(x), t2, t3))
    return rows


@case("rope", "D · RoPE：v2 index_select+chunk vs v3 Triton 单核")
def _rope(args):
    v2, v3 = make_rope("v2"), make_rope("v3")
    rows = []
    for tokens, heads in [(4096, 16), (256, 16), (32, 16)]:
        pos = torch.arange(tokens, device="cuda")
        q = torch.randn(tokens, heads, 128, **D)
        k = torch.randn(tokens, 8, 128, **D)
        t2 = bench(v2, pos, q, k, warm=args.warm, iters=args.iters)
        t3 = bench(v3, pos, q, k, warm=args.warm, iters=args.iters)
        rows.append((f"{tokens}×{heads}×128", t2, t3))
    return rows


@case("silu", "D · SiLU+Mul：v2 两核 vs v3 FlashInfer 单核")
def _silu(args):
    v2, v3 = make_silu("v2"), make_silu("v3")
    rows = []
    for shape in [(1, 4096, 6144), (8, 256, 6144), (32, 1, 6144)]:
        x = torch.randn(*shape, **D)
        t2 = bench(v2, x, warm=args.warm, iters=args.iters)
        t3 = bench(v3, x, warm=args.warm, iters=args.iters)
        rows.append((shape_of(x), t2, t3))
    return rows


@case("mlp", "D · MLP：v2 双 GEMM+cat vs v3 融合 GEMM")
def _mlp(args):
    v2, v3 = make_mlp("v2"), make_mlp("v3")
    rows = []
    for shape in [(1, 4096, 1024), (8, 256, 1024), (32, 1, 1024)]:
        x = torch.randn(*shape, **D)
        t2 = bench(v2, x, warm=args.warm, iters=args.iters)
        t3 = bench(v3, x, warm=args.warm, iters=args.iters)
        rows.append((shape_of(x), t2, t3))
    return rows


@case("kvwrite", "C · KV 落盘：v2 index_copy_ vs v3 Triton 单核")
def _kvwrite(args):
    v2, v3 = make_kvwrite("v2"), make_kvwrite("v3")
    rows = []
    for n in [32, 256, 4096]:
        k = torch.randn(n, 8, 128, **D)
        v = torch.randn_like(k)
        slots = torch.arange(n, device="cuda")
        t2 = bench(v2, k, v, slots, warm=args.warm, iters=args.iters)
        t3 = bench(v3, k, v, slots, warm=args.warm, iters=args.iters)
        rows.append((f"{n} tok", t2, t3))
    return rows


def make_attn_prefill_fa2(num_heads=16, num_kv_heads=8, head_dim=128):
    """v3 首 chunk prefill：FlashAttention-2 varlen（v3 use_fa2 主路径）。"""
    from flash_attn import flash_attn_func

    def run(q, k, v):
        # q/k/v: [T, H, D]
        out = flash_attn_func(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), causal=True
        )
        return out[0].contiguous()

    return run


@case("attn_prefill", "A · prefill attention：v2 SDPA vs v3 FA2 / FlashInfer")
def _attn_prefill(args):
    v2 = make_attn_prefill_v2()
    v3_fa2 = make_attn_prefill_fa2()
    v3_flashinf = make_attn_prefill_v3()
    rows = []
    for seq_len in [256, 1024, 4096]:
        q = torch.randn(seq_len, 16, 128, **D)
        k = torch.randn(seq_len, 8, 128, **D)
        v = torch.randn_like(k)
        t2 = bench(v2, q, k, v, warm=args.warm, iters=args.iters)
        t3f = bench(v3_fa2, q, k, v, warm=args.warm, iters=args.iters)
        t3i = bench(v3_flashinf, q, seq_len, warm=args.warm, iters=args.iters)
        rows.append((f"1×{seq_len}×16×128", t2, t3f, t3i))
    return rows


@case("attn_decode", "A · decode attention：v2 逐条 SDPA vs v3 FlashInfer batch")
def _attn_decode(args):
    v2 = make_attn_decode_v2()
    v3 = make_attn_decode_v3()
    rows = []
    for bs in [1, 8, 32]:
        for ctx in [256, 4096]:
            q = torch.randn(bs, 16, 128, **D)
            t2 = bench(v2, q, ctx, warm=args.warm, iters=args.iters)
            t3 = bench(v3, q, ctx, bs, warm=args.warm, iters=args.iters)
            rows.append((f"{bs}×1×16×128 ctx={ctx}", t2, t3))
    return rows


@case("sampler", "E · Sampler：v2 sort+multinomial vs v3 topk+gumbel")
def _sampler(args):
    v2, v3 = make_sampler("v2"), make_sampler("v3")
    rows = []
    for bs in [1, 8, 256]:
        logits = torch.randn(bs, 151936, **D)
        temp = torch.full((bs,), 0.9, **D)
        top_p = torch.full((bs,), 0.95, **D)
        top_k = torch.full((bs,), 20, dtype=torch.int32, device="cuda")
        t2 = bench(v2, logits, temp, top_p, top_k,
                   warm=args.warm, iters=args.iters)
        t3 = bench(v3, logits, temp, top_ps=top_p, top_ks=top_k,
                   warm=args.warm, iters=args.iters)
        rows.append((f"{bs}×151936", t2, t3))
    return rows


@case("layer_decode", "整层 · decode 前向（含 qkv/o 投影，eager 组合路径）")
def _layer_decode(args):
    v2_attn = make_attn_decode_v2()
    v3_attn = make_attn_decode_v3()
    v2_mlp, v3_mlp = make_mlp("v2"), make_mlp("v3")
    v2_ln, v3_ln = make_rmsnorm("v2"), make_rmsnorm("v3")

    qkv2 = torch.nn.Linear(1024, (16 + 2 * 8) * 128, bias=False).to(**D)
    qkv3 = torch.nn.Linear(1024, (16 + 2 * 8) * 128, bias=False).to(**D)
    o2 = torch.nn.Linear(2048, 1024, bias=False).to(**D)
    o3 = torch.nn.Linear(2048, 1024, bias=False).to(**D)

    def v2_layer(x, ctx):
        h = v2_ln(x)                     # [bs,1,1024]
        qkv = qkv2(h)                    # [bs,1,4096]
        q, _, _ = qkv.split([2048, 1024, 1024], dim=-1)
        q = q.reshape(-1, 16, 128)
        o = v2_attn(q, ctx)              # [bs,16,128]
        o = o.reshape(-1, 2048)
        o = o2(o)
        return v2_mlp(v2_ln(o.reshape_as(x)))

    def v3_layer(x, ctx, bs):
        h = v3_ln(x)
        qkv = qkv3(h)
        q, _, _ = qkv.split([2048, 1024, 1024], dim=-1)
        q = q.reshape(-1, 16, 128)
        o = v3_attn(q, ctx, bs)
        o = o.reshape(-1, 2048)
        o = o3(o)
        return v3_mlp(v3_ln(o.reshape_as(x)))

    rows = []
    for bs in [1, 8, 32]:
        ctx = 4096
        x = torch.randn(bs, 1, 1024, **D)
        t2 = bench(v2_layer, x, ctx, warm=args.warm, iters=args.iters)
        t3 = bench(v3_layer, x, ctx, bs, warm=args.warm, iters=args.iters)
        rows.append((f"{bs} tok decode, ctx={ctx}", t2, t3))
    return rows


@case("layer_prefill", "整层 · prefill 前向（含 qkv/o 投影，eager 组合路径）")
def _layer_prefill(args):
    v2_attn = make_attn_prefill_v2()
    v3_attn = make_attn_prefill_v3()
    v2_mlp, v3_mlp = make_mlp("v2"), make_mlp("v3")
    v2_ln, v3_ln = make_rmsnorm("v2"), make_rmsnorm("v3")

    qkv2 = torch.nn.Linear(1024, (16 + 2 * 8) * 128, bias=False).to(**D)
    qkv3 = torch.nn.Linear(1024, (16 + 2 * 8) * 128, bias=False).to(**D)
    o2 = torch.nn.Linear(2048, 1024, bias=False).to(**D)
    o3 = torch.nn.Linear(2048, 1024, bias=False).to(**D)

    def v2_layer(x):
        h = v2_ln(x)                     # [1,T,1024]
        qkv = qkv2(h)                    # [1,T,4096]
        q, k, v = qkv.split([2048, 1024, 1024], dim=-1)
        q = q.reshape(-1, 16, 128)
        k = k.reshape(-1, 8, 128)
        v = v.reshape(-1, 8, 128)
        o = v2_attn(q, k, v)             # [T,16,128]
        o = o.reshape(-1, 2048)
        o = o2(o)
        return v2_mlp(v2_ln(o.reshape_as(x)))

    def v3_layer(x, seq_len):
        h = v3_ln(x)
        qkv = qkv3(h)
        q, _, _ = qkv.split([2048, 1024, 1024], dim=-1)
        q = q.reshape(-1, 16, 128)
        o = v3_attn(q, seq_len)
        o = o.reshape(-1, 2048)
        o = o3(o)
        return v3_mlp(v3_ln(o.reshape_as(x)))

    rows = []
    for seq_len in [256, 1024, 4096]:
        x = torch.randn(1, seq_len, 1024, **D)
        t2 = bench(v2_layer, x, warm=args.warm, iters=args.iters)
        t3 = bench(v3_layer, x, seq_len, warm=args.warm, iters=args.iters)
        rows.append((f"1×{seq_len}×1024", t2, t3))
    return rows


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def fmt_row(name: str, label: str, rows) -> str:
    ncols = max(len(r) for r in rows) - 1  # v3 实现列数（2 或 3）
    lines = [f"\n### {name} — {label}", ""]
    if ncols == 2:
        lines += [
            "| 输入形状 | v2 (ms) | v3 (ms) | 加速比 |",
            "| --- | ---: | ---: | ---: |",
        ]
        for shape, t2, t3 in rows:
            ratio = t2 / t3 if t3 > 0 else float("inf")
            lines.append(f"| {shape} | {t2:.3f} | {t3:.3f} | {ratio:.2f}x |")
    else:
        lines += [
            "| 输入形状 | v2 SDPA (ms) | v3 FA2 (ms) | v3 FlashInfer (ms) | FA2 加速 | FI 加速 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for shape, t2, t3f, t3i in rows:
            lines.append(
                f"| {shape} | {t2:.3f} | {t3f:.3f} | {t3i:.3f} | "
                f"{t2 / t3f:.2f}x | {t2 / t3i:.2f}x |"
            )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="", help="逗号分隔的模块名，空=全部")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warm", type=int, default=50)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    D["device"] = args.device

    names = list(CASES) if not args.only else args.only.split(",")
    print(f"# v2 vs v3 算子级性能剖析（{args.device}）\n")
    md = [f"# v2 vs v3 算子级性能剖析", ""]
    for name in names:
        rows = CASES[name]["fn"](args)
        block = fmt_row(name, CASES[name]["label"], rows)
        print(block)
        md.append(block)
    out = Path(args.device.replace(":", "_") + "_ops_bench.md")
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / out.name
    out.write_text("\n".join(md) + "\n")
    print(f"\n→ saved to {out}")


if __name__ == "__main__":
    main()
