"""vllm-v2 基准入口。

用法::

    # 推荐：多请求真实 Prefill 显存对比
    python -m bench.run_all --suite kv_vram --model Qwen3-1.7B --context-len 4096

    # PagedAttention 相关（kv_vram + paged_overhead）
    python -m bench.run_all --suite paged --model Qwen3-1.7B --context-len 4096

    # 全部套件
    python -m bench.run_all --suite all --model Qwen3-0.6B --context-len 2048

``kv_vram`` 会对 R 路请求真实 Prefill+Decode；连续对照为 R×M 估算（见输出脚注）。
"""

from __future__ import annotations

import argparse

from bench.lib.cli import add_common_args, config_from_args
from bench.lib.config import BenchConfig, DEFAULT_LENGTH_SWEEP
from bench.lib.registry import ALL_SUITES, PAGED_SUITES, SUITE_REGISTRY
from bench.lib.report import ResultWriter
from bench.lib.session import BenchSession, collect_meta
from bench.suites import kv_vram as kv_vram_suite


def resolve_suites(name: str) -> list[str]:
    if name == "all":
        return ALL_SUITES
    if name == "paged":
        return list(PAGED_SUITES)
    if name not in SUITE_REGISTRY:
        raise SystemExit(f"未知套件: {name}，可选: paged|all|{','.join(ALL_SUITES)}")
    return [name]


def run_suites(cfg: BenchConfig, suite_names: list[str]) -> ResultWriter:
    print(f"加载模型 {cfg.model} …")
    meta = collect_meta(cfg)
    writer = ResultWriter(cfg, meta)

    # kv_vram 需要按场景扩池；若同一次 run 里有它，先算好 num_gpu_blocks。
    if "kv_vram" in suite_names and cfg.num_gpu_blocks is None:
        cfg.num_gpu_blocks = kv_vram_suite.needed_gpu_blocks(cfg)

    own_session = {"latency_seqlen"}
    shared = [n for n in suite_names if n not in own_session]
    session: BenchSession | None = None
    if shared:
        session = BenchSession(cfg)
        if any(n != "kv_vram" for n in shared):
            session.global_warmup()

    for name in suite_names:
        print(f"\n>>> 运行套件: {name}")
        if name in own_session:
            SUITE_REGISTRY[name](cfg, writer, session=None)
        else:
            assert session is not None
            SUITE_REGISTRY[name](cfg, writer, session=session)

    print(f"\n完成。报告: {writer.run_dir / 'report.md'}")
    return writer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="vllm-v2 基准（PagedAttention）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "套件: kv_vram(多请求真实Prefill) | paged_overhead | ttft | prefill | "
            "decode | e2e | vram | latency_seqlen\n"
            "组合: --suite paged (=kv_vram+paged_overhead) | --suite all"
        ),
    )
    add_common_args(parser)
    parser.add_argument(
        "--suite",
        default="kv_vram",
        help="套件名，或 paged / all（默认 kv_vram）",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=DEFAULT_LENGTH_SWEEP,
        help="prefill/latency/paged_overhead 扫描长度",
    )
    parser.add_argument(
        "--paged-lengths",
        type=int,
        nargs="+",
        default=None,
        help="仅跑一组自定义并发长度（传给 kv_vram，跳过内置场景）",
    )
    parser.add_argument("--prompt-len", type=int, default=BenchConfig().latency_prompt_len)
    parser.add_argument("--decode-steps", type=int, default=None)
    parser.add_argument("--sample-every", type=int, default=BenchConfig().latency_sample_every)
    parser.set_defaults(max_new_tokens=64)
    args = parser.parse_args()

    cfg = config_from_args(args)
    cfg.prefill_lengths = args.lengths
    cfg.latency_lengths = args.lengths
    cfg.latency_prompt_len = args.prompt_len
    if args.paged_lengths is not None:
        cfg.paged_lengths = args.paged_lengths
    cfg.latency_decode_steps = (
        args.decode_steps
        if args.decode_steps is not None
        else cfg.context_len - args.prompt_len
    )

    run_suites(cfg, resolve_suites(args.suite))


if __name__ == "__main__":
    main()
