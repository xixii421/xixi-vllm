"""Stage 3 — GPU model step (worker-local).

``ModelRunner.run``: build tensors → (optional) vision encode → forward → sample.
``mrope_sync`` bridges worker snapshots back to the engine ``Sequence`` state.
"""

from .model_inputs import ModelInputs, build_model_inputs
from .model_runner import ModelRunner, ensure_single_process_parallel
from .mrope_sync import (
    MRopeStateUpdate,
    apply_mrope_updates,
    snapshot_mrope_updates,
)
from .sampler import Sampler

__all__ = [
    "ModelInputs",
    "ModelRunner",
    "MRopeStateUpdate",
    "Sampler",
    "apply_mrope_updates",
    "build_model_inputs",
    "ensure_single_process_parallel",
    "snapshot_mrope_updates",
]
