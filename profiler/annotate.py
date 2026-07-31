from __future__ import annotations

import functools
from typing import Callable
from torch.profiler import record_function

def profile(name: str | None = None) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with record_function(name or fn.__name__):
                return fn(*args, **kwargs)
        return wrapper

    return decorator