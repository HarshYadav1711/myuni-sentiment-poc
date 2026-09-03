"""Temporal reasoner model-comparison harness (Phase 3B-A).

Isolates LLM quality from upstream variance by freezing deterministic
TemporalContext payloads. Does not download models in unit tests.
Does not embed ZeroGPU deployment into core temporal logic.
"""

from src.temporal.benchmark.capabilities import (
    MODEL_CANDIDATES,
    ModelCapability,
    resolve_model_capability,
)
from src.temporal.benchmark.export import (
    FrozenReasonerPayload,
    export_frozen_payload,
    lean_temporal_context_for_reasoner,
    load_frozen_payload,
)
from src.temporal.benchmark.schemas import (
    HumanReviewFields,
    InvariantCheckResult,
    ReasonerBenchmarkResult,
)

__all__ = [
    "MODEL_CANDIDATES",
    "FrozenReasonerPayload",
    "HumanReviewFields",
    "InvariantCheckResult",
    "ModelCapability",
    "ReasonerBenchmarkResult",
    "export_frozen_payload",
    "lean_temporal_context_for_reasoner",
    "load_frozen_payload",
    "resolve_model_capability",
]
