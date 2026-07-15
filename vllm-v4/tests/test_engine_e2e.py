"""End-to-end Engine checks (eager + CUDA Graph) and admission edge cases."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

from src import Engine, EngineConfig, SamplingParams


MODEL = Path.home() / "huggingface" / "Qwen3-0.6B"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not MODEL.is_dir(),
    reason="needs CUDA + Qwen3-0.6B",
)


@pytest.fixture(scope="module")
def eng_eager():
    eng = Engine(
        EngineConfig(
            model=str(MODEL),
            enforce_eager=True,
            max_num_seqs=4,
            context_len=512,
            num_kvcache_blocks=32,
        )
    )
    yield eng
    eng.model_runner.destroy()


@pytest.fixture(scope="module")
def eng_cg():
    eng = Engine(
        EngineConfig(
            model=str(MODEL),
            enforce_eager=False,
            max_num_seqs=4,
            context_len=512,
            num_kvcache_blocks=32,
        )
    )
    yield eng
    eng.model_runner.destroy()


class TestEngineE2E:
    def test_greedy_batch_eager(self, eng_eager):
        outs = eng_eager.generate_batch(
            ["The capital of France is", "1+1="],
            SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True),
        )
        assert len(outs) == 2
        assert all(isinstance(o, str) and len(o) > 0 for o in outs)

    def test_eager_matches_cg_greedy(self, eng_eager, eng_cg):
        prompts = ["Hello, my name is", "Once upon a time"]
        sp = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
        a = eng_eager.generate_batch(prompts, sp)
        b = eng_cg.generate_batch(prompts, sp)
        assert a == b

    def test_context_clamp_no_rope_oob(self, eng_eager):
        # Fill the window exactly; max_tokens must clamp to 0 — no extra position.
        context = eng_eager.context_len
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            seq = eng_eager.add_request(
                list(range(context)),
                SamplingParams(temperature=0.0, max_tokens=8),
            )
        assert seq.max_tokens == 0
        assert seq.num_prompt_tokens == context
        assert any("clamping max_tokens" in str(x.message) for x in w)
        # Drain without generating
        while not eng_eager.scheduler.is_finished():
            eng_eager.step()
        assert seq.is_finished
        assert seq.num_completion_tokens == 0

    def test_cg_reserves_dummy_page(self, eng_cg):
        phys = eng_cg.kv_cache.shape[2]
        assert eng_cg.scheduler.block_manager.num_blocks == phys - 1
        assert eng_cg.model_runner.cuda_graph_runner is not None
        assert eng_cg.model_runner.cuda_graph_runner.dummy_page_id == phys - 1

    def test_cg_requires_two_blocks(self):
        with pytest.raises(RuntimeError, match=">= 2 KV blocks"):
            Engine(
                EngineConfig(
                    model=str(MODEL),
                    enforce_eager=False,
                    max_num_seqs=2,
                    context_len=128,
                    num_kvcache_blocks=1,
                )
            )

    def test_stream_generate(self, eng_eager):
        toks = list(
            eng_eager.generate(
                "Hello",
                max_new_tokens=4,
                temperature=0.0,
            )
        )
        assert 0 < len(toks) <= 4
