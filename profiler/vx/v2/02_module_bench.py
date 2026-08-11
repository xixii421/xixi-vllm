"""v2 各子模块的微基准剖析（对齐 v3 瓶颈 A~E）。

每个模块独立构造输入、独立跑 profile，输出到 out/：
  - {module}_trace.json    Chrome trace，模块名已通过 record_function 写入，
                           Perfetto 打开后按模块名过滤即见该模块的全部算子
  - {module}_profile.txt   精简算子表（仅 5 列，Top 行）

模块与 v3 瓶颈对照：
  attention  → A gather+SDPA        kvwrite  → C index_copy_
  rmsnorm    → D pow/mean/rsqrt      rope     → D index_select+chunk
  mlp        → D 多算子碎片          sampler  → E 多次 sort + 同步
  decode     → B 整步启动开销（对照基准）

用法:
    python 02_module_bench.py --module all          # 全部模块
    python 02_module_bench.py --module attention    # 单模块
    python 02_module_bench.py --iters 50 --warm 10
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "vllm-v2"))

from src.config import EngineConfig  # noqa: E402
from src.engine import Engine  # noqa: E402
from src.kv import ForwardContext, set_forward_context  # noqa: E402
from src.model_hub import DEFAULT_CACHE_DIR  # noqa: E402
from src.sampling_params import SamplingParams  # noqa: E402

BENCHES: dict[str, tuple[str, object]] = {}


def register(name: str, label: str):
    def deco(fn):
        BENCHES[name] = (label, fn)
        return fn

    return deco


# ---------------------------------------------------------------------------
# 模块注册：每个 prepare(engine, args) 返回一个无参 step()
# ---------------------------------------------------------------------------


@register("attention", "A · qkv_proj + rotary + gather + SDPA")
def prepare_attention(engine, args):
    attn = engine.model.model.layers[0].self_attn  # Qwen3Attention
    store = engine.kv.store
    n_kv, hd = store.pool.num_kv_heads, store.pool.head_dim
    seq_len = 64

    # 预写一层 KV（64 个 slot），让 decode gather 有东西可读
    k_full = torch.randn(seq_len, n_kv, hd, device=engine.device, dtype=engine.dtype)
    v_full = torch.randn_like(k_full)
    store.write(0, torch.arange(seq_len, device=engine.device), k_full, v_full)

    slots = torch.tensor([seq_len - 1], device=engine.device)
    ctx = ForwardContext(
        is_prefill=False,
        store=store,
        block_tables=[[0, 1, 2, 3]],
        slot_mapping=slots,
        query_lens=[1],
        context_lens=[seq_len],
    )
    hidden = torch.randn(
        1, 1, attn.qkv_proj.weight.shape[1], device=engine.device, dtype=engine.dtype
    )
    positions = torch.tensor([[seq_len - 1]], device=engine.device)

    def step():
        with record_function("attention"):
            set_forward_context(ctx)
            attn(positions, hidden)
            set_forward_context(None)

    return step


@register("kvwrite", "C · index_copy_")
def prepare_kvwrite(engine, args):
    store = engine.kv.store
    n_kv, hd = store.pool.num_kv_heads, store.pool.head_dim
    n = 256  # 模拟 256 个 token 批量写入
    k = torch.randn(n, n_kv, hd, device=engine.device, dtype=engine.dtype)
    v = torch.randn_like(k)
    slots = torch.arange(n, device=engine.device)

    def step():
        with record_function("kvwrite"):
            store.write(0, slots, k, v)

    return step


@register("rmsnorm", "D · pow/mean/rsqrt 三段")
def prepare_rmsnorm(engine, args):
    ln = engine.model.model.layers[0].input_layernorm
    x = torch.randn(8, 128, ln.weight.numel(), device=engine.device, dtype=engine.dtype)

    def step():
        with record_function("rmsnorm"):
            ln(x)

    return step


@register("rope", "D · index_select + chunk + rotate")
def prepare_rope(engine, args):
    rot = engine.model.model.layers[0].self_attn.rotary_emb
    hd = rot.head_size
    positions = torch.arange(128, device=engine.device)
    q = torch.randn(1, 128, rot.head_size * 2, device=engine.device, dtype=engine.dtype)
    k = torch.randn(1, 128, rot.head_size, device=engine.device, dtype=engine.dtype)

    def step():
        with record_function("rope"):
            rot(positions, q, k)

    return step


@register("mlp", "D · 4 算子碎片")
def prepare_mlp(engine, args):
    mlp = engine.model.model.layers[0].mlp
    x = torch.randn(
        8, 128, mlp.gate_proj.weight.shape[1], device=engine.device, dtype=engine.dtype
    )

    def step():
        with record_function("mlp"):
            mlp(x)

    return step


@register("sampler", "E · sort + softmax + multinomial")
def prepare_sampler(engine, args):
    sampler = engine.sampler
    vocab = engine.model.lm_head.weight.shape[0]
    logits = torch.randn(8, vocab, device=engine.device, dtype=engine.dtype)
    temps = torch.tensor([0.9] * 8, device=engine.device)
    top_p = torch.tensor([0.95] * 8, device=engine.device)
    top_k = torch.tensor([20] * 8, device=engine.device)

    def step():
        with record_function("sampler"):
            sampler(logits, temps, top_ps=top_p, top_ks=top_k)

    return step


@register("decode", "B · 整步启动开销（对照基准）")
def prepare_decode(engine, args):
    base = engine.tokenizer.encode("The quick brown fox jumps over the lazy dog.")
    tokens = (base * 4)[: args.prompt_len]
    seq = engine.add_request(tokens, SamplingParams(temperature=0.0, max_tokens=64))
    for _ in range(4):  # 进入 decode 稳态
        engine.step()

    def step():
        with record_function("decode"):
            engine.step()

    return step


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def compact_table(prof, row_limit=15) -> str:
    """精简算子表：只保留 Name / Calls / CPU self / CUDA / CUDA%，宽度可控。"""
    evs = sorted(prof.key_averages(), key=lambda e: e.device_time_total, reverse=True)
    total = sum(e.device_time_total for e in evs) or 1.0
    lines = [f"{'#':>3} {'Calls':>6} {'CPU self ms':>11} {'CUDA ms':>9} "
             f"{'CUDA%':>7}  Name", "-" * 76]
    for i, e in enumerate(evs[:row_limit]):
        name = (e.key or "?").split("(")[0][:44]
        lines.append(
            f"{i:3d} {e.count:6d} {e.self_cpu_time_total / 1e6:11.3f} "
            f"{e.device_time_total / 1e6:9.3f} "
            f"{e.device_time_total / total * 100:6.1f}%  {name}"
        )
    return "\n".join(lines)


def run_bench(name: str, label: str, prepare, engine, args) -> None:
    step = prepare(engine, args)
    for _ in range(args.warm):
        step()
    torch.cuda.synchronize()

    schedule = torch.profiler.schedule(wait=0, warmup=0, active=args.iters, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule,
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(args.iters):
            step()
            prof.step()
    torch.cuda.synchronize()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = compact_table(prof, args.row_limit)
    print(f"\n===== {name}  ({label}) =====")
    print(table)
    (out / f"{name}_profile.txt").write_text(
        table + f"\n\nSelf CPU: {sum(e.self_cpu_time_total for e in prof.key_averages()) / 1e6:.1f}ms  "
        f"Self CUDA: {sum(e.device_time_total for e in prof.key_averages()) / 1e6:.1f}ms"
    )
    prof.export_chrome_trace(str(out / f"{name}_trace.json"))
    print(f"  → {out / name}_profile.txt + {name}_trace.json")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--module", choices=["all"] + list(BENCHES), default="all")
    p.add_argument("--prompt-len", type=int, default=128, help="decode 对照用 prompt 长度")
    p.add_argument("--iters", type=int, default=50, help="profile 记录步数")
    p.add_argument("--warm", type=int, default=10, help="预热步数")
    p.add_argument("--row-limit", type=int, default=12, help="算子表行数")
    p.add_argument("--out-dir", default="./out")
    return p.parse_args()


def main():
    args = parse_arguments()
    engine = Engine(
        EngineConfig(
            model=str(DEFAULT_CACHE_DIR / "Qwen3-0.6B"),
            context_len=2048,
            block_size=16,
            prefix_backend="hash",
            max_num_seqs=8,
            max_num_batched_tokens=4096,
            mix_prefill_decode=True,
        )
    )
    names = list(BENCHES) if args.module == "all" else [args.module]
    for name in names:
        label, prepare = BENCHES[name]
        run_bench(name, label, prepare, engine, args)


if __name__ == "__main__":
    main()
