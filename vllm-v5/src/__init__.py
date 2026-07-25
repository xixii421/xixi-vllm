from __future__ import annotations
from typing import TYPE_CHECKING, Any
__all__ = ['Engine', 'EngineConfig', 'SamplingParams']
if TYPE_CHECKING:
    from .config import EngineConfig as EngineConfig
    from .runtime import Engine as Engine
    from .sampling import SamplingParams as SamplingParams

def __getattr__(name: str) -> Any:
    if name == 'EngineConfig':
        from .config import EngineConfig
        return EngineConfig
    if name == 'Engine':
        from .runtime import Engine
        return Engine
    if name == 'SamplingParams':
        from .sampling import SamplingParams
        return SamplingParams
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
