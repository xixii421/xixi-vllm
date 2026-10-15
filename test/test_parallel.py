"""v4 并行正确性验证：多卡（TP/PP）greedy 输出对拍。

用法:
    python test/test_parallel.py --tp 1 --pp 1                    # 单卡
    python test/test_parallel.py --tp 2 --pp 1                    # 机内 TP（需 >=2 张空闲卡）
    python test/test_parallel.py --tp 1 --pp 2                    # PP 跨卡
    python test/test_parallel.py --tp 2 --pp 2                    # TP+PP（需 4 张卡）
    python test/test_parallel.py --model ~/huggingface/Qwen3-8B --tp 1 --pp 2

对拍基准：默认 Qwen3-0.6B 用 bf16 HF 参考（GOLDEN，关闭低精度 reduction 与
v4 的 matmul 精度一致）；其它模型只打印输出，由调用方对比 pp=1 与 pp=2 的逐
token 一致性（纯层切分应 bit 级一致）。

多机 torchrun 路径见 run_torchrun.py（rank0 兼调度器）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import Engine, EngineConfig, SamplingParams  # noqa: E402

MODEL = str(Path.home() / "huggingface" / "Qwen3-0.6B")

# bf16 HF 参考输出（Qwen3-0.6B，prompt_len=64, salt=7, temperature=0, max_tokens=16,
# ignore_eos；关闭 allow_bf16_reduced_precision_reduction 以匹配 v4 精度）。
GOLDEN = [271, 40, 1184, 311, 1477, 279, 1372, 315, 56015, 315, 279, 3409, 330, 1782, 1, 304]


def make_cfg(
    tp: int,
    pp: int,
    *,
    model: str = MODEL,
    num_blocks: int = 32,
    device_ids: list[int] | None = None,
    pp_layer_counts: list[int] | None = None,
    enforce_eager: bool = True,
    torch_compile: bool = False,
) -> EngineConfig:
    return EngineConfig(
        model=model,
        tp_size=tp,
        pp_size=pp,
        enforce_eager=enforce_eager,
        torch_compile=torch_compile,
        dtype="bfloat16",
        context_len=512,
        max_num_seqs=16,
        max_num_batched_tokens=1024,
        prefix_backend="none",
        gpu_memory_utilization=0.6,
        num_kvcache_blocks=num_blocks,
        device_ids=device_ids,
        pp_layer_counts=pp_layer_counts,
    )


def dummy_ids(n: int, salt: int = 0) -> list[int]:
    n = max(1, n)
    return [((i * 7919 + 10007 + salt * 97) % 100000) + 100 for i in range(n)]


def run_greedy(
    tp: int,
    pp: int,
    *,
    model: str = MODEL,
    prompt_len: int = 64,
    max_tokens: int = 16,
    device_ids: list[int] | None = None,
    pp_layer_counts: list[int] | None = None,
    enforce_eager: bool = True,
    torch_compile: bool = False,
) -> list[int]:
    engine = Engine(
        make_cfg(
            tp,
            pp,
            model=model,
            device_ids=device_ids,
            pp_layer_counts=pp_layer_counts,
            enforce_eager=enforce_eager,
            torch_compile=torch_compile,
        )
    )
    try:
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        seq = engine.add_request(dummy_ids(prompt_len, salt=7), sp)
        while not seq.is_finished:
            engine.step()
        return seq.token_ids[seq.num_prompt_tokens:]
    finally:
        engine.destroy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL, help="模型路径（默认 Qwen3-0.6B）")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument(
        "--device-ids",
        type=str,
        default=None,
        help="逗号分隔的设备映射（如 0,0 把两个 rank 都放到 GPU 0，仅调试，NCCL 不允许同卡多 rank）",
    )
    p.add_argument(
        "--pp-layer-counts",
        type=str,
        default=None,
        help="不均分 PP 每 stage 层数，逗号分隔（如 12,28）；默认等分",
    )
    p.add_argument(
        "--cg",
        action="store_true",
        help="启用 decode piecewise CUDA Graph（PP 时 recv/send 在图外）",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="启用 prefill torch.compile",
    )
    args = p.parse_args()

    device_ids = None
    if args.device_ids is not None:
        device_ids = [int(x) for x in args.device_ids.split(",")]

    pp_layer_counts = None
    if args.pp_layer_counts is not None:
        pp_layer_counts = [int(x) for x in args.pp_layer_counts.split(",")]

    model = str(Path(args.model).expanduser())

    out = run_greedy(
        args.tp,
        args.pp,
        model=model,
        prompt_len=args.prompt_len,
        max_tokens=args.max_tokens,
        device_ids=device_ids,
        pp_layer_counts=pp_layer_counts,
        enforce_eager=not args.cg,
        torch_compile=args.compile,
    )
    # 默认 0.6B 且并行配置时对拍 bf16 golden；其它模型仅打印，由调用方对比 pp=1/pp=2。
    is_06b = Path(model).name == "Qwen3-0.6B"
    use_golden = is_06b and (args.tp > 1 or args.pp > 1) and args.prompt_len == 64
    if use_golden:
        golden = GOLDEN[: args.max_tokens]
        ok = out == golden
        print(f"tp={args.tp} pp={args.pp} {'PASS' if ok else 'FAIL'} completion={out}")
        sys.exit(0 if ok else 1)
    print(f"tp={args.tp} pp={args.pp} completion={out}")


if __name__ == "__main__":
    main()
