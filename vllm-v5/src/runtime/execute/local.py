"""In-process executor for ``world_size == 1`` (zero IPC / pickle)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...config import EngineConfig
from ...scheduling import Sequence

if TYPE_CHECKING:
    from ..runner.model_runner import ModelRunner


class _CompletedFuture:
    """Minimal future matching ``ModelExecutionFuture.result()`` usage."""

    __slots__ = ("_result",)

    def __init__(self, result: tuple[list[int], list | None]) -> None:
        self._result = result

    def done(self) -> bool:
        return True

    def result(self, timeout: float | None = None) -> tuple[list[int], list | None]:
        del timeout
        return self._result


class LocalExecutor:
    """Engine-facing executor that runs ``ModelRunner`` in the same process."""

    def __init__(self, config: EngineConfig, device: torch.device):
        from ..runner.model_runner import ModelRunner

        self.config = config
        self.model_runner: ModelRunner = ModelRunner(config, device)

    def submit_execute_model(
        self, seqs: list[Sequence], is_prefill: bool
    ) -> _CompletedFuture:
        token_ids, mrope_updates = self.model_runner.run(seqs, is_prefill)
        assert isinstance(token_ids, list)
        return _CompletedFuture((token_ids, mrope_updates))

    def get_num_kvcache_blocks(self) -> int:
        return self.model_runner.num_blocks

    def reset_sys_profile(self) -> None:
        from ...utils.profiler import reset

        reset()

    def get_sys_profile(self) -> list[dict]:
        from ...utils.profiler import summary

        return [{**summary(), "_rank": 0}]

    def shutdown(self) -> None:
        self.model_runner.destroy()
