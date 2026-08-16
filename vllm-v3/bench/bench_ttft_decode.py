"""规范口径：TTFT / TPOT / prefix hit。

对齐 vLLM bench serve 定义（docs.vllm.ai/benchmarking/cli）：

- TTFT = 请求开始 → 第一个 completion token（CUDA synchronize）
- TPOT = (E2E − TTFT) / (n_out − 1)   # 不含首 token 的 decode
- decode tok/s = 1000 / TPOT_ms
- 并发 = 1（不含排队）；warmup 3 次后再取 n 次中位数

对照设置：
- v3:  decode CG + compile + **radix**
- nano: CUDA Graph + 默认 **hash** 前缀缓存
- 官方 vLLM: 默认 **hash** APC（enable_prefix_caching=True）

prefix 场景对齐 benchmark_prefix_caching / Long Document QA：
公共前缀 1024 token（块对齐 256），独特后缀 128，先 miss 写入再测 hit。

用法（在 vllm-v3/ 下，三引擎分进程）:
    ../.venv2/bin/python bench/bench_ttft_decode.py --engine v3
    ../.venv2/bin/python bench/bench_ttft_decode.py --engine nano
    ../.venv-vllm/bin/python bench/bench_ttft_decode.py --engine vllm
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

BENCH = Path(__file__).resolve().parent
V3_ROOT = BENCH.parent
REPO = V3_ROOT.parent
NANO_ROOT = REPO / "nano-vllm"
OUT = BENCH / "out"
MODEL = str(Path.home() / "huggingface" / "Qwen3-0.6B")

TEMP = 0.6
WARMUP = 3
N_SAMPLES = 8
DECODE_TOKENS = 64
ISOLATED_LENS = (256, 1024)
PREFIX_LEN = 1024  # 256 块对齐，v3/nano 都能整块命中
SUFFIX_LEN = 128


def dummy_ids(n: int, salt: int = 0) -> list[int]:
    n = max(1, n)
    return [((i * 7919 + 10007 + salt * 97) % 150000) + 100 for i in range(n)]


def median(xs: list[float]) -> float:
    ys = sorted(xs)
    return ys[len(ys) // 2]


def _tpot_ms(e2e_ms: float, ttft_ms: float, n_out: int) -> float:
    if n_out <= 1:
        return 0.0
    return (e2e_ms - ttft_ms) / (n_out - 1)


class Runner:
    tag: str = ""
    prefix_name: str = ""

    def warmup(self, prompt: list[int], max_tokens: int = 4) -> None:
        raise NotImplementedError

    def timed(self, prompt: list[int], max_tokens: int) -> dict:
        """返回 ttft_ms, e2e_ms, n_out, cached。"""
        raise NotImplementedError

    def close(self) -> None:
        pass


class V3Runner(Runner):
    tag = "decodeCG+compile"
    prefix_name = "radix"

    def __init__(self, device: str) -> None:
        if str(BENCH) not in sys.path:
            sys.path.insert(0, str(BENCH))
        if str(V3_ROOT) not in sys.path:
            sys.path.insert(0, str(V3_ROOT))
        from common import destroy, make_engine  # noqa: E402
        from src import SamplingParams  # noqa: E402

        self._destroy = destroy
        self._SP = SamplingParams
        torch.cuda.set_device(device)
        self.engine = make_engine(
            cg=True, compile=True, prefix="radix", max_num_seqs=8, gpu_mem=0.85
        )

    def warmup(self, prompt: list[int], max_tokens: int = 4) -> None:
        sp = self._SP(temperature=TEMP, max_tokens=max_tokens, ignore_eos=True)
        for it in range(WARMUP):
            seq = self.engine.add_request(dummy_ids(len(prompt), salt=7000 + it), sp)
            while not seq.is_finished:
                self.engine.step()
        torch.cuda.synchronize()

    def timed(self, prompt: list[int], max_tokens: int) -> dict:
        sp = self._SP(temperature=TEMP, max_tokens=max_tokens, ignore_eos=True)
        seq = self.engine.add_request(prompt, sp)
        cached = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ttft_s = None
        while not seq.is_finished:
            self.engine.step()
            if cached == 0:
                cached = int(self.engine.last_num_cached_tokens)
            if ttft_s is None and seq.num_completion_tokens > 0:
                torch.cuda.synchronize()
                ttft_s = time.perf_counter() - t0
        torch.cuda.synchronize()
        e2e = (time.perf_counter() - t0) * 1000
        ttft = (ttft_s or (e2e / 1000)) * 1000
        return {
            "ttft_ms": ttft,
            "e2e_ms": e2e,
            "n_out": seq.num_completion_tokens,
            "cached": cached,
        }

    def close(self) -> None:
        self._destroy(self.engine)


class NanoRunner(Runner):
    tag = "CUDA Graph + hash"
    prefix_name = "hash"

    def __init__(self) -> None:
        if str(NANO_ROOT) not in sys.path:
            sys.path.insert(0, str(NANO_ROOT))
        from nanovllm import LLM, SamplingParams  # noqa: E402

        self._SP = SamplingParams
        self.llm = LLM(
            MODEL,
            enforce_eager=False,
            max_model_len=4096,
            max_num_seqs=8,
            gpu_memory_utilization=0.85,
        )

    def _seqs(self):
        return list(self.llm.scheduler.waiting) + list(self.llm.scheduler.running)

    def warmup(self, prompt: list[int], max_tokens: int = 4) -> None:
        sp = self._SP(temperature=TEMP, max_tokens=max_tokens, ignore_eos=True)
        for it in range(WARMUP):
            self.llm.generate(
                [dummy_ids(len(prompt), salt=7000 + it)], sp, use_tqdm=False
            )
        torch.cuda.synchronize()

    def timed(self, prompt: list[int], max_tokens: int) -> dict:
        sp = self._SP(temperature=TEMP, max_tokens=max_tokens, ignore_eos=True)
        self.llm.add_request(prompt, sp)
        seq0 = self.llm.scheduler.waiting[-1]
        n_hit_blocks = self.llm.scheduler.block_manager.can_allocate(seq0)
        cached = max(0, int(n_hit_blocks)) * int(seq0.block_size)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ttft_s = None
        n_out = 0
        while not self.llm.is_finished():
            self.llm.step()
            for seq in self._seqs():
                if ttft_s is None and seq.num_completion_tokens > 0:
                    torch.cuda.synchronize()
                    ttft_s = time.perf_counter() - t0
                n_out = max(n_out, seq.num_completion_tokens)
        torch.cuda.synchronize()
        e2e = (time.perf_counter() - t0) * 1000
        ttft = (ttft_s or (e2e / 1000)) * 1000
        return {"ttft_ms": ttft, "e2e_ms": e2e, "n_out": n_out, "cached": cached}

    def close(self) -> None:
        try:
            self.llm.exit()
        except Exception:
            pass


class VllmRunner(Runner):
    tag = "v0.26 hash APC"
    prefix_name = "hash"

    def __init__(self) -> None:
        from vllm import LLM, SamplingParams  # noqa: E402

        try:
            from vllm import TokensPrompt  # noqa: E402

            self._prompt = lambda ids: TokensPrompt(prompt_token_ids=ids)
        except Exception:
            self._prompt = lambda ids: dict(prompt_token_ids=ids)
        self._SP = SamplingParams
        self.llm = LLM(
            model=MODEL,
            enforce_eager=False,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            max_num_seqs=8,
            enable_prefix_caching=True,
        )

    def warmup(self, prompt: list[int], max_tokens: int = 4) -> None:
        sp = self._SP(temperature=TEMP, max_tokens=max_tokens, ignore_eos=True)
        for it in range(WARMUP):
            self.llm.generate(
                [self._prompt(dummy_ids(len(prompt), salt=7000 + it))],
                sp,
                use_tqdm=False,
            )
        torch.cuda.synchronize()

    def timed(self, prompt: list[int], max_tokens: int) -> dict:
        """离线 LLM.generate 没有可靠流式时间戳：先 max_tokens=1 测 TTFT，
        再用同一 prompt 跑 N token（APC 已命中）近似纯 decode。
        """
        sp1 = self._SP(temperature=TEMP, max_tokens=1, ignore_eos=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        o1 = self.llm.generate([self._prompt(prompt)], sp1, use_tqdm=False)[0]
        torch.cuda.synchronize()
        ttft = (time.perf_counter() - t0) * 1000
        cached = int(o1.num_cached_tokens or 0)

        spn = self._SP(temperature=TEMP, max_tokens=max_tokens, ignore_eos=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        o2 = self.llm.generate([self._prompt(prompt)], spn, use_tqdm=False)[0]
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t1) * 1000
        n_out = len(o2.outputs[0].token_ids)
        tpot = decode_ms / n_out if n_out else 0.0
        return {
            "ttft_ms": ttft,
            "e2e_ms": ttft + decode_ms,
            "n_out": n_out,
            "cached": cached,
            "_tpot_ms": tpot,
        }


def _summarize(samples: list[dict]) -> dict:
    ttfts = [s["ttft_ms"] for s in samples]
    tpots = []
    for s in samples:
        if "_tpot_ms" in s and s["_tpot_ms"] > 0:
            tpots.append(s["_tpot_ms"])
        else:
            tpots.append(_tpot_ms(s["e2e_ms"], s["ttft_ms"], s["n_out"]))
    tpot = median(tpots)
    return {
        "ttft_ms": median(ttfts),
        "tpot_ms": tpot,
        "decode_tps": 1000.0 / tpot if tpot > 0 else 0.0,
        "cached": int(median([float(s["cached"]) for s in samples])),
        "n": len(samples),
    }


def run_isolated(runner: Runner) -> dict:
    out = {}
    for L in ISOLATED_LENS:
        prompt = dummy_ids(L, salt=1)
        print(f"  isolated prompt={L}: warmup {WARMUP} …", flush=True)
        runner.warmup(prompt, max_tokens=4)
        samples = []
        for i in range(N_SAMPLES):
            samples.append(
                runner.timed(dummy_ids(L, salt=10 + i), max_tokens=DECODE_TOKENS)
            )
        out[str(L)] = _summarize(samples)
        r = out[str(L)]
        print(
            f"    TTFT {r['ttft_ms']:.2f} ms  TPOT {r['tpot_ms']:.3f} ms  "
            f"decode {r['decode_tps']:.1f} tok/s",
            flush=True,
        )
    return out


def run_prefix(runner: Runner) -> dict:
    prefix = dummy_ids(PREFIX_LEN, salt=42)
    print(f"  prefix miss/hit  prefix={PREFIX_LEN} suffix={SUFFIX_LEN} …", flush=True)
    runner.warmup(prefix[:256] + dummy_ids(32, salt=0), max_tokens=4)

    miss_prompt = prefix + dummy_ids(SUFFIX_LEN, salt=0)
    miss = runner.timed(miss_prompt, max_tokens=DECODE_TOKENS)
    hits = []
    for i in range(1, N_SAMPLES + 1):
        hits.append(
            runner.timed(prefix + dummy_ids(SUFFIX_LEN, salt=i), max_tokens=DECODE_TOKENS)
        )
    hit = _summarize(hits)
    miss_tpot = miss.get("_tpot_ms") or _tpot_ms(miss["e2e_ms"], miss["ttft_ms"], miss["n_out"])
    rec = {
        "prefix_len": PREFIX_LEN,
        "suffix_len": SUFFIX_LEN,
        "miss_ttft_ms": miss["ttft_ms"],
        "miss_cached": miss["cached"],
        "miss_tpot_ms": miss_tpot,
        "hit_ttft_ms": hit["ttft_ms"],
        "hit_cached": hit["cached"],
        "hit_tpot_ms": hit["tpot_ms"],
        "hit_decode_tps": hit["decode_tps"],
        "ttft_speedup": (miss["ttft_ms"] / hit["ttft_ms"]) if hit["ttft_ms"] else 0.0,
    }
    print(
        f"    miss TTFT {rec['miss_ttft_ms']:.2f} ms cached={rec['miss_cached']} | "
        f"hit TTFT {rec['hit_ttft_ms']:.2f} ms cached={rec['hit_cached']} "
        f"({rec['ttft_speedup']:.2f}x)  hit decode {rec['hit_decode_tps']:.1f} tok/s",
        flush=True,
    )
    return rec


def _append_section(path: Path, header: str, lines: list[str]) -> None:
    """追加一节到 txt；header 仅在文件中尚未出现时写一次。"""
    OUT.mkdir(parents=True, exist_ok=True)
    text = path.read_text() if path.exists() else ""
    with path.open("a") as f:
        if header not in text:
            f.write(header + "\n")
        for ln in lines:
            f.write(ln + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=("v3", "nano", "vllm"), required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    print(f"# engine={args.engine}  (TTFT / TPOT / prefix)\n", flush=True)

    if args.engine == "v3":
        runner: Runner = V3Runner(args.device)
    elif args.engine == "nano":
        runner = NanoRunner()
    else:
        runner = VllmRunner()

    engine = args.engine
    prefix = runner.prefix_name
    try:
        isolated = run_isolated(runner)
        prefix_hit = run_prefix(runner)
    finally:
        runner.close()

    iso_lines = [
        f"{engine} | {prefix} | {L} | {s['ttft_ms']:.2f} | "
        f"{s['tpot_ms']:.3f} | {s['decode_tps']:.1f}"
        for L, s in isolated.items()
    ]
    _append_section(
        OUT / "bench_ttft_decode.txt",
        "engine | prefix | prompt | TTFT (ms) | TPOT (ms) | decode tok/s",
        iso_lines,
    )

    pf = prefix_hit
    pf_lines = [
        f"{engine} | {pf['miss_ttft_ms']:.2f} | {pf['hit_ttft_ms']:.2f} | "
        f"{pf['ttft_speedup']:.2f}x | {pf['hit_cached']} | {pf['hit_decode_tps']:.1f}"
    ]
    _append_section(
        OUT / "bench_ttft_decode.txt",
        "engine | miss TTFT (ms) | hit TTFT (ms) | speedup | hit cached | hit decode tok/s",
        pf_lines,
    )
    print(f"→ {OUT / 'bench_ttft_decode.txt'}", flush=True)


if __name__ == "__main__":
    main()
