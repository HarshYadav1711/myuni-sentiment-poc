"""Temporal reasoner provider protocol."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from src.schemas import (
    SentimentEvidence,
    TemporalContext,
    TemporalReasonerDiagnostics,
    TemporalReasoningResult,
)


@runtime_checkable
class TemporalReasonerProvider(Protocol):
    """Shared interface for Qwen and OpenRouter temporal reasoners."""

    @property
    def model_id(self) -> str: ...

    def reason(
        self,
        temporal: TemporalContext,
        *,
        baseline_overall: Optional[SentimentEvidence] = None,
    ) -> tuple[TemporalReasoningResult, TemporalReasonerDiagnostics]: ...
