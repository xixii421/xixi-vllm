"""Benchmark session helpers for v3 FlashInfer engine."""

from __future__ import annotations

import gc

from dataclasses import dataclass

import torch

from src import Engine, EngineConfig, SamplingParams
from src.memory import Sequence
from src.utils import reset_context

from .config import BenchConfig
from .timing import CudaTimer, sync


@dataclass(frozen=True)
class BenchMeta:
    model: str
    gpu: str
    vram_gb: float
    torch_version: str
    cuda_version: str
    enforce_eager: bool
    warmup_iters: int
    measure_iters: int
    block_size: int = 256
    num_kvcache_blocks: int = 0


def collect_meta(cfg: BenchConfig) -> BenchMeta:
    gpu = "cpu"
    vram = 0.0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = torch.cuda.get_device_name(0)
        vram = props.total_memory / (1024**3)
    return BenchMeta(
        model=cfg.model,
        gpu=gpu,
        vram_gb=round(vram, 2),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "n/a",
        enforce_eager=cfg.enforce_eager,
        warmup_iters=cfg.warmup_iters,
        measure_iters=cfg.measure_iters,
        block_size=cfg.block_size,
        num_kvcache_blocks=cfg.num_kvcache_blocks or 0,
    )


class BenchSession:
    def __init__(self, cfg: BenchConfig) -> None:
        if not cfg.model_path.is_dir():
            raise FileNotFoundError(
                f"Model not found at {cfg.model_path}. Run: python run.py --download"
            )
        self.cfg = cfg
        ecfg = EngineConfig(
            model=str(cfg.model_path),
            context_len=cfg.context_len,
            kvcache_block_size=cfg.block_size,
            max_num_seqs=cfg.max_num_seqs,
            num_kvcache_blocks=cfg.num_kvcache_blocks or 0,
            enforce_eager=cfg.enforce_eager,
        )
        self.engine = Engine(ecfg)
        self.device = self.engine.device
        self._bench_seq: Sequence | None = None
        self._held_seqs: list[Sequence] = []

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    @property
    def block_manager(self):
        return self.engine.kv.blocks

    @property
    def kv(self):
        return self.engine.kv

    @property
    def model_runner(self):
        return self.engine.model_runner

    def global_warmup(self, steps: int = 3) -> None:
        for _ in range(steps):
            list(
                self.engine.generate(
                    "Hello",
                    max_new_tokens=4,
                    temperature=0.0,
                )
            )
        sync()

    def make_prompt_tokens(self, target_len: int) -> list[int]:
        chunk = "The quick brown fox jumps over the lazy dog. "
        ids: list[int] = []
        while len(ids) < target_len:
            ids.extend(self.tokenizer.encode(chunk, add_special_tokens=False))
        return ids[:target_len]

    def clear_kv_state(self) -> None:
        if self._bench_seq is not None and self._bench_seq.block_table:
            self.kv.deallocate(self._bench_seq, publish=False)
        self._bench_seq = None
        for seq in self._held_seqs:
            if seq.block_table:
                self.kv.deallocate(seq, publish=False)
        self._held_seqs = []
        reset_context()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        sync()

    def _free_bench_seq(self) -> None:
        if self._bench_seq is not None and self._bench_seq.block_table:
            self.kv.deallocate(self._bench_seq, publish=False)
        self._bench_seq = None

    @torch.inference_mode()
    def prefill_forward_only(self, input_ids: list[int]) -> None:
        self._free_bench_seq()
        seq = Sequence.from_prompt(list(input_ids), block_size=self.engine.block_size)
        assert self.kv.can_allocate(seq)
        self.kv.allocate(seq)
        self._bench_seq = seq
        start = seq.num_cached_tokens
        end = len(input_ids)
        seq.num_scheduled_tokens = end - start
        self.model_runner.run([seq], [])
        seq.num_cached_tokens = end
        seq.num_scheduled_tokens = 0

    @torch.inference_mode()
    def decode_forward_only(self, cache_len: int, next_token_id: int = 1) -> None:
        if self._bench_seq is None:
            raise RuntimeError("decode_forward_only requires prefill first")
        seq = self._bench_seq
        seq.append_token(next_token_id)
        self.kv.may_append(seq)
        seq.num_scheduled_tokens = 1
        self.model_runner.run([], [seq])
        seq.num_scheduled_tokens = 0

    def fill_kv_cache(self, prompt_ids: list[int]) -> None:
        self.prefill_forward_only(prompt_ids)

    def multi_request_prefill(
        self, lengths: list[int], *, decode_steps: int = 1
    ) -> dict:
        self.clear_kv_state()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        seqs: list[Sequence] = []
        self._held_seqs = seqs
        bs = self.engine.block_size
        for t_len in lengths:
            ids = self.make_prompt_tokens(t_len)
            seq = Sequence.from_prompt(ids, block_size=bs)
            if not self.kv.can_allocate(seq):
                raise RuntimeError("block pool exhausted during multi_request_prefill")
            self.kv.allocate(seq)
            seqs.append(seq)
            start = seq.num_cached_tokens
            end = len(ids)
            seq.num_scheduled_tokens = end - start
            token = self.model_runner.run([seq], [])[0][0]
            self.kv.sync_prefix(seq)
            seq.num_cached_tokens = end
            seq.num_scheduled_tokens = 0
            if decode_steps > 0:
                seq.append_token(token)
                self.kv.may_append(seq)
                seq.num_scheduled_tokens = 1
                self.model_runner.run([], [seq])
                seq.num_scheduled_tokens = 0
        sync()
        used_blocks = sum(len(s.block_table) for s in seqs)
        return {
            "num_requests": len(lengths),
            "used_blocks": used_blocks,
            "free_blocks": self.kv.num_free,
            "peak_gb": round(self.peak_memory_gb(), 4),
            "allocated_gb": round(self.allocated_memory_gb(), 4),
        }

    def forward_cuda_ms(self, input_ids: list[int]) -> float:
        with CudaTimer() as timer:
            self.prefill_forward_only(input_ids)
        return timer.ms

    def forward_only(self, input_ids: list[int]) -> None:
        self.prefill_forward_only(input_ids)

    def greedy_step_cuda_ms(self, input_ids: list[int]) -> float:
        self.fill_kv_cache(input_ids)

        def _run() -> None:
            self.decode_forward_only(len(input_ids))

        with CudaTimer() as timer:
            _run()
        return timer.ms

    def reset_peak_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def peak_memory_gb(self) -> float:
        sync()
        return torch.cuda.max_memory_allocated() / (1024**3)

    def allocated_memory_gb(self) -> float:
        sync()
        return torch.cuda.memory_allocated() / (1024**3)

    def generate_batch_cb(
        self, prompts: list[str], max_tokens: int = 32
    ) -> list[str]:
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        return self.engine.generate_batch(prompts, sp)
