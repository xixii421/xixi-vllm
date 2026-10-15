"""检查命令行预热与单条 / 批量路径的资源释放。"""

from __future__ import annotations

import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run  # noqa: E402


class EntrypointCleanupTests(unittest.TestCase):
    def test_warmup_finishes_before_measurement(self) -> None:
        engine = Mock()
        pending = []
        engine.add_request.side_effect = lambda *args: pending.append(args)
        engine.has_unfinished.side_effect = lambda: bool(pending)
        engine.step.side_effect = lambda: pending.pop()
        with patch.object(run, "cuda_sync"):
            run._warmup(engine, [2, 3])
        self.assertEqual(pending, [])
        self.assertGreater(engine.step.call_count, 0)
        self.assertEqual(engine.step.call_count, engine.add_request.call_count)

    def check_cleanup(self, entrypoint, failure: str | None = None) -> None:
        engine = Mock()
        engine.tokenizer.encode.return_value = [1]
        engine.tokenizer.decode.return_value = "text"
        engine.add_request.return_value = SimpleNamespace(
            is_finished=failure != "generation",
            token_ids=[1, 2],
            num_prompt_tokens=1,
            num_completion_tokens=1,
        )
        engine.has_unfinished.return_value = failure == "generation"
        engine.step.side_effect = RuntimeError("generation failed")
        with ExitStack() as stack:
            stack.enter_context(patch.object(run, "_make_engine", return_value=engine))
            warmup = stack.enter_context(patch.object(run, "_warmup", return_value=0.0))
            stack.enter_context(patch.object(run, "cuda_sync"))
            stack.enter_context(patch.object(run, "print_bench_report"))
            stack.enter_context(patch("builtins.print"))
            if failure == "warmup":
                warmup.side_effect = RuntimeError("warmup failed")
            if failure is None:
                entrypoint(dict(run.DEFAULTS))
            else:
                with self.assertRaisesRegex(RuntimeError, f"{failure} failed"):
                    entrypoint(dict(run.DEFAULTS))
        engine.destroy.assert_called_once_with()

    def test_success(self) -> None:
        for entrypoint in (run.run_single, run.run_batch):
            with self.subTest(entrypoint=entrypoint.__name__):
                self.check_cleanup(entrypoint)

    def test_warmup_failure(self) -> None:
        for entrypoint in (run.run_single, run.run_batch):
            with self.subTest(entrypoint=entrypoint.__name__):
                self.check_cleanup(entrypoint, "warmup")

    def test_generation_failure(self) -> None:
        for entrypoint in (run.run_single, run.run_batch):
            with self.subTest(entrypoint=entrypoint.__name__):
                self.check_cleanup(entrypoint, "generation")


if __name__ == "__main__":
    unittest.main()
