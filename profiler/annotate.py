from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from typing import Any

import torch


def _current_profiler() -> Any | None:
    try:
        return torch.profiler._utils._HOOKS
    except Exception:
        return None


class _DummyRange:
    def __enter__(self): return self
    def __exit__(self, *args): return False


def profile_scope(name: str):
    try:
        return torch.profiler.record_function(name)
    except Exception:
        return _DummyRange()


def profile_range(name: str):
    def _decorator(fn: Callable) -> Callable:
        def _wrapper(*args, **kwargs):
            with profile_scope(name):
                return fn(*args, **kwargs)
        _wrapper.__name__ = fn.__name__
        _wrapper.__qualname__ = fn.__qualname__
        _wrapper.__doc__ = fn.__doc__
        return _wrapper
    return _decorator
