"""Runtime package — lazy façade (avoid import cycles with cuda_graph).

Call order: Engine → input → schedule → execute → runner (+ vision).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..scheduling.scheduler import Scheduler
    from .engine import Engine
    from .runner import ModelRunner

__all__ = ["Engine", "ModelRunner", "Scheduler"]


def __getattr__(name: str) -> Any:
    if name == "Engine":
        from .engine import Engine

        return Engine
    if name == "ModelRunner":
        from .runner import ModelRunner

        return ModelRunner
    if name == "Scheduler":
        from ..scheduling.scheduler import Scheduler

        return Scheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
