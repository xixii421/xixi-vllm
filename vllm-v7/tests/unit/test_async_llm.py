"""AsyncLLM continuous-batching façade (no GPU)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.runtime.async_llm import AsyncLLM
from src.sampling import SamplingParams
from src.scheduling.sequence import Sequence, SequenceStatus


class _FakeTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return ",".join(str(t) for t in token_ids)


class _FakeScheduler:
    def __init__(self) -> None:
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self.aborted: list[int] = []

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def abort(self, seq: Sequence) -> None:
        self.aborted.append(seq.seq_id)
        seq.status = SequenceStatus.FINISHED
        if seq in self.waiting:
            self.waiting.remove(seq)
        if seq in self.running:
            self.running.remove(seq)


class _FakeEngine:
    """Deterministic engine: each step appends one token to every running seq."""

    def __init__(self) -> None:
        self.scheduler = _FakeScheduler()
        self.tokenizer = _FakeTokenizer()
        self.destroyed = False
        self.steps = 0
        self.max_parallel = 0

    def add_request(
        self, prompt=None, sampling_params=None, images=None, *, chat_messages=None
    ):
        del images, chat_messages
        if isinstance(prompt, list):
            tokens = list(prompt)
        elif prompt is None:
            tokens = [1, 2, 3]
        else:
            tokens = [1, 2, 3]
        seq = Sequence(tokens, sampling_params, block_size=16)
        self.scheduler.add(seq)
        return seq

    def abort(self, seq: Sequence) -> None:
        self.scheduler.abort(seq)

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    def decode_tokens(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def step(self):
        self.steps += 1
        while self.scheduler.waiting:
            seq = self.scheduler.waiting.pop(0)
            seq.status = SequenceStatus.RUNNING
            self.scheduler.running.append(seq)

        self.max_parallel = max(self.max_parallel, len(self.scheduler.running))
        finished = []
        for seq in list(self.scheduler.running):
            next_id = 1000 + seq.num_completion_tokens
            seq.append_token(next_id)
            if seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.scheduler.running.remove(seq)
                finished.append(seq)
        return ([(s.seq_id, s.token_ids[s.num_prompt_tokens :]) for s in finished], 0)

    def begin_step(self):
        return object() if not self.scheduler.is_finished() else None

    def wait_step(self, pending):
        del pending
        return None

    def finish_step(self, pending, result):
        del pending, result
        return self.step()

    def destroy(self) -> None:
        self.destroyed = True


def _async_llm_with_fake_engine() -> tuple[AsyncLLM, _FakeEngine]:
    eng = _FakeEngine()
    llm = object.__new__(AsyncLLM)
    llm.config = SimpleNamespace()
    llm._engine = eng
    llm._lock = asyncio.Lock()
    llm._requests = {}
    llm._has_work = asyncio.Event()
    llm._stopped = False
    llm._loop_task = None
    llm._abort_ids = set()
    return llm, eng


def test_concurrent_generate_batches_requests():
    async def _run():
        llm, eng = _async_llm_with_fake_engine()
        sp = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)

        async def consume(prompt: list[int]) -> list[int]:
            out = []
            async for chunk in llm.generate(prompt, sp):
                if chunk["token_id"] is not None:
                    out.append(chunk["token_id"])
            return out

        results = await asyncio.gather(
            consume([10, 11]),
            consume([20, 21]),
            consume([30, 31]),
        )
        await llm.shutdown()
        return results, eng

    results, eng = asyncio.run(_run())
    assert results == [[1000, 1001, 1002, 1003]] * 3
    # Without the old global generate-lock, one step can advance multiple seqs.
    assert eng.max_parallel >= 2
    # 3 requests × 4 tokens would take ≥12 serialized steps; batching is less.
    assert eng.steps < 12
    assert eng.destroyed


def test_abort_on_cancel_frees_request():
    async def _run():
        llm, eng = _async_llm_with_fake_engine()
        sp = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)

        async def slow():
            async for _ in llm.generate([1, 2, 3], sp):
                await asyncio.sleep(0)

        task = asyncio.create_task(slow())
        for _ in range(200):
            if eng.scheduler.running or eng.scheduler.waiting or llm._requests:
                break
            await asyncio.sleep(0.001)
        assert llm._requests or eng.scheduler.running or eng.scheduler.waiting
        task.cancel()
        try:
            await task
            raise AssertionError("expected CancelledError")
        except asyncio.CancelledError:
            pass
        # Background loop applies non-blocking abort enqueue.
        for _ in range(200):
            if eng.scheduler.aborted and not llm._requests:
                break
            await asyncio.sleep(0.001)
        aborted = list(eng.scheduler.aborted)
        pending = dict(llm._requests)
        await llm.shutdown()
        return aborted, pending

    aborted, pending = asyncio.run(_run())
    assert aborted
    assert not pending


def test_generate_full_joins_text():
    async def _run():
        llm, _eng = _async_llm_with_fake_engine()
        text = await llm.generate_full(
            [1, 2],
            SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True),
        )
        await llm.shutdown()
        return text

    assert asyncio.run(_run()) == "100010011002"
