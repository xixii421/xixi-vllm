"""vllm-v3 统一入口 — FlashAttention + CUDA Graph + Continuous Batching。

用法:
    python run.py                                    # 单条推理（使用 config.yaml）
    python run.py --eager                            # 强制 eager（关闭 Decode CUDA Graph）
    python run.py --cg                               # 启用 Decode CUDA Graph
    python run.py --compile                          # 启用 prefill torch.compile
    python run.py --no-compile                       # 关闭 prefill torch.compile
    python run.py --batch                            # 批量推理演示
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from src import Engine, EngineConfig, SamplingParams
from hub import (
    DEFAULT_CACHE_DIR,
    QWEN3_REGISTRY,
    download_models,
    gpu_memory_gb,
    select_models_for_vram,
)

# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent

DEFAULTS = {
    "model": "Qwen3-0.6B",
    "prompt": "Hello, my name is",
    "max_new_tokens": 200,
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 20,
    "enforce_eager": True,
    "torch_compile": False,
    "compile_mode": "reduce-overhead",
    "compile_dynamic": True,
    "dtype": "auto",
    "context_len": 4096,
    "max_num_seqs": 256,
    "max_num_batched_tokens": 8192,
    "mix_prefill_decode": True,
    "kvcache_block_size": 256,
    "prefix_backend": "hash",
    "gpu_memory_utilization": 0.9,
    "num_kvcache_blocks": 0,
    "tp_size": 1,
    "pp_size": 1,
    "dp_size": 1,
    "pp_layer_counts": None,
    "nccl_port": 29500,
    "master_addr": "127.0.0.1",
    "master_port": 29500,
    "nccl_ifname": "",
}


def load_config(path: Path) -> dict:
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    return {k: cfg.get(k, v) for k, v in DEFAULTS.items()}


# ---------------------------------------------------------------------------
# 核心逻辑
# ---------------------------------------------------------------------------


def _resolve_model(name: str) -> Path:
    path = Path(name).expanduser()
    return path if path.exists() else DEFAULT_CACHE_DIR / name


def _make_engine(cfg: dict) -> Engine:
    model_path = _resolve_model(cfg["model"])
    print(f"Loading {model_path} ...")
    return Engine(
        EngineConfig(
            model=str(model_path),
            enforce_eager=cfg["enforce_eager"],
            torch_compile=cfg["torch_compile"],
            compile_mode=cfg["compile_mode"],
            compile_dynamic=cfg["compile_dynamic"],
            dtype=cfg["dtype"],
            context_len=cfg["context_len"],
            max_num_seqs=cfg["max_num_seqs"],
            max_num_batched_tokens=cfg["max_num_batched_tokens"],
            mix_prefill_decode=cfg["mix_prefill_decode"],
            kvcache_block_size=cfg["kvcache_block_size"],
            prefix_backend=cfg["prefix_backend"],
            gpu_memory_utilization=cfg["gpu_memory_utilization"],
            num_kvcache_blocks=cfg["num_kvcache_blocks"],
            tp_size=cfg["tp_size"],
            pp_size=cfg["pp_size"],
            dp_size=cfg["dp_size"],
            pp_layer_counts=cfg["pp_layer_counts"],
            nccl_port=cfg["nccl_port"],
            master_addr=cfg["master_addr"],
            master_port=cfg["master_port"],
            nccl_ifname=cfg["nccl_ifname"],
        )
    )


def _sync() -> None:
    torch.cuda.synchronize()


def _drain(engine: Engine) -> None:
    while engine.has_unfinished():
        engine.step()


# reduce-overhead 对同一形状：1) 编译 2) CUDA Graph 录制 3) 重放。
# 预热必须把前两次付完，计时才是稳态 TTFT。
_WARMUP_ITERS = 3


def _dummy_ids(n: int, salt: int = 0) -> list[int]:
    """与真实 prompt 等长、每轮内容不同，避免前缀命中把预热/TTFT 短路。"""
    n = max(1, n)
    return [((i * 7919 + 10007 + salt * 97) % 100000) + 100 for i in range(n)]


def _warmup(engine: Engine, prompt_lens: list[int]) -> float:
    """把本形状的 compile + 录图跑完，不计入后续 TTFT / Decode / E2E。

    多条 lens 会作为同一批 prefill（对齐 --batch 的合批形状）。
    不用 reset_kv（会拆 runner 并重新编译）。
    """
    warm_sp = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)
    _sync()
    t0 = time.perf_counter()
    for it in range(_WARMUP_ITERS):
        for n in prompt_lens:
            engine.add_request(_dummy_ids(n, salt=it * 31 + n), warm_sp)
        _drain(engine)
    _sync()
    return time.perf_counter() - t0


def _report(
    ttft_s: float | None,
    n: int,
    e2e_s: float,
    *,
    decode_s: float | None = None,
    warmup_s: float | None = None,
) -> None:
    """TTFT / Decode 均为预热后的稳态数字；warmup 单独打印。"""
    print()
    if warmup_s is not None:
        print(f"Warmup: {warmup_s:.2f}s  (compile + CUDA Graph capture, excluded)")
    if ttft_s is not None:
        print(f"TTFT:   {ttft_s * 1000:.1f} ms  (steady prefill + first token)")
    n_dec = max(0, n - 1)
    if decode_s is not None and decode_s > 0 and n_dec > 0:
        print(f"Decode: {n_dec} tokens in {decode_s:.3f}s ({n_dec / decode_s:.1f} tok/s)")
    elif n_dec == 0:
        print("Decode: (no tokens after the first; raise max_new_tokens)")
    if e2e_s > 0:
        print(f"E2E:    {n} tokens in {e2e_s:.3f}s ({n / e2e_s:.1f} tok/s)")


def run_single(cfg: dict) -> None:
    """单条推理：等长 dummy 先编译再录图，第三条起才计时。"""
    engine = _make_engine(cfg)
    prompt = cfg["prompt"]
    token_ids = (
        list(prompt) if isinstance(prompt, list) else engine.tokenizer.encode(prompt)
    )
    warmup_s = _warmup(engine, [len(token_ids)])

    sp = SamplingParams(
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
        max_tokens=cfg["max_new_tokens"],
    )
    seq = engine.add_request(token_ids, sp)

    _sync()
    t0 = time.perf_counter()
    t_first = None
    while not seq.is_finished:
        engine.step()
        if t_first is None and seq.num_completion_tokens > 0:
            _sync()
            t_first = time.perf_counter()
    _sync()
    t_end = time.perf_counter()

    completion = seq.token_ids[seq.num_prompt_tokens :]
    text = engine.tokenizer.decode(completion)
    print(f"Prompt: {prompt}")
    print(f"{prompt}{text}")
    _report(
        (t_first - t0) if t_first is not None else None,
        len(completion),
        t_end - t0,
        decode_s=(t_end - t_first) if t_first is not None else None,
        warmup_s=warmup_s,
    )


def run_batch(cfg: dict) -> None:
    """批量推理：按各 prompt 等长 dummy 预热后计时。"""
    engine = _make_engine(cfg)
    prompts = [
        "Hello, my name is",
        "Once upon a time",
        "The capital of France is",
        "In machine learning,",
    ]
    encoded = [engine.tokenizer.encode(p) for p in prompts]
    warmup_s = _warmup(engine, [len(ids) for ids in encoded])

    sp = SamplingParams(temperature=0.0, max_tokens=cfg["max_new_tokens"])
    seqs = [engine.add_request(ids, sp) for ids in encoded]

    _sync()
    t0 = time.perf_counter()
    t_first = None
    while engine.has_unfinished():
        engine.step()
        if t_first is None and any(s.num_completion_tokens > 0 for s in seqs):
            _sync()
            t_first = time.perf_counter()
    _sync()
    t_end = time.perf_counter()

    n = 0
    for p, seq in zip(prompts, seqs):
        ids = seq.token_ids[seq.num_prompt_tokens :]
        n += len(ids)
        print(f"\n>>> {p}\n{engine.tokenizer.decode(ids)}")
    _report(
        (t_first - t0) if t_first is not None else None,
        n,
        t_end - t0,
        decode_s=(t_end - t_first) if t_first is not None else None,
        warmup_s=warmup_s,
    )
    print(f"Batch:  {len(prompts)} reqs, {n} completion tokens")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="vllm-v3 FlashAttention + CUDA Graph + Continuous Batching"
    )
    p.add_argument("--config", default=None, help="YAML 配置文件路径")
    p.add_argument("--model", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    cg_group = p.add_mutually_exclusive_group()
    cg_group.add_argument("--cg", action="store_true", help="启用 Decode CUDA Graph")
    cg_group.add_argument(
        "--eager", action="store_true", help="强制 eager 模式（关闭 Decode CUDA Graph）"
    )
    compile_group = p.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--compile", action="store_true", help="启用 prefill torch.compile"
    )
    compile_group.add_argument(
        "--no-compile", action="store_true", help="关闭 prefill torch.compile"
    )
    p.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"),
                   default=None)
    p.add_argument("--compile-dynamic", action="store_true",
                   help="符号形状（一次编译覆盖任意长度，避免按长度重编译）")
    p.add_argument("--no-compile-dynamic", action="store_true",
                   help="固定形状重放（仅离线固定长度；新长度会重编译）")
    p.add_argument("--prefix-backend", choices=("none", "hash", "radix"), default=None)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--tp", type=int, default=None, help="张量并行度（机内多卡）")
    p.add_argument("--pp", type=int, default=None, help="流水线并行度（跨机/多卡）")
    p.add_argument("--dp", type=int, default=None, help="数据并行副本数（多卡吞吐）")
    p.add_argument(
        "--pp-layer-counts",
        type=str,
        default=None,
        help="不均分 PP 每 stage 层数（逗号分隔，如 12,28）",
    )
    p.add_argument("--batch", action="store_true", help="运行批量推理演示")
    p.add_argument("--download", action="store_true")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    # 管理命令
    if args.list:
        vram = gpu_memory_gb() or 24.0
        selected = select_models_for_vram(vram_gb=vram)
        print(f"Cache: {DEFAULT_CACHE_DIR}")
        print(f"VRAM:  {vram:.1f} GB")
        print(f"All:   {', '.join(s.local_name for s in QWEN3_REGISTRY)}")
        print(f"Fit:   {', '.join(s.local_name for s in selected)}")
        return
    if args.download:
        print(f"Downloading to {DEFAULT_CACHE_DIR} ...")
        download_models(list(QWEN3_REGISTRY))
        print("Done.")
        return

    config_path = Path(args.config) if args.config else HERE / "config.yaml"
    cfg = load_config(config_path)
    for key in ("model", "prompt", "max_new_tokens", "temperature", "top_p", "top_k",
                "prefix_backend", "max_num_seqs"):
        if (v := getattr(args, key)) is not None:
            cfg[key] = v
    if args.tp is not None:
        cfg["tp_size"] = args.tp
    if args.pp is not None:
        cfg["pp_size"] = args.pp
    if args.dp is not None:
        cfg["dp_size"] = args.dp
    if args.pp_layer_counts is not None:
        cfg["pp_layer_counts"] = [int(x) for x in args.pp_layer_counts.split(",")]
    if args.cg:
        cfg["enforce_eager"] = False
    if args.eager:
        cfg["enforce_eager"] = True
    if args.compile:
        cfg["torch_compile"] = True
    if args.no_compile:
        cfg["torch_compile"] = False
    if args.compile_mode:
        cfg["compile_mode"] = args.compile_mode
    if args.compile_dynamic:
        cfg["compile_dynamic"] = True
    if args.no_compile_dynamic:
        cfg["compile_dynamic"] = False

    if args.batch:
        run_batch(cfg)
    else:
        run_single(cfg)


if __name__ == "__main__":
    main()
