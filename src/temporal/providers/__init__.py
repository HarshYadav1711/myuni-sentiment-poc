"""Factory for temporal reasoner providers (OpenRouter / Qwen)."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional, Union

from src.config import DEFAULT_TEMPORAL_REASONER, TemporalReasonerConfig
from src.schemas import (
    SentimentEvidence,
    TemporalContext,
    TemporalReasonerDiagnostics,
    TemporalReasoningResult,
)
from src.temporal.providers.base import TemporalReasonerProvider
from src.temporal.providers.openrouter import OpenRouterTemporalReasoner
from src.temporal.reasoner import TemporalContextReasoner

logger = logging.getLogger(__name__)

ReasonerImpl = Union[OpenRouterTemporalReasoner, TemporalContextReasoner]


class FallbackTemporalReasoner:
    """Optional explicit fallback wrapper (default: no automatic Qwen fallback)."""

    def __init__(
        self,
        primary: ReasonerImpl,
        *,
        fallback: Optional[ReasonerImpl] = None,
        fallback_name: str = "none",
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.fallback_name = fallback_name

    @property
    def model_id(self) -> str:
        return self.primary.model_id

    def reason(
        self,
        temporal: TemporalContext,
        *,
        baseline_overall: Optional[SentimentEvidence] = None,
    ) -> tuple[TemporalReasoningResult, TemporalReasonerDiagnostics]:
        result, diagnostics = self.primary.reason(
            temporal,
            baseline_overall=baseline_overall,
        )
        if result.status == "ok":
            return result, diagnostics
        if self.fallback is None or self.fallback_name == "none":
            return result, diagnostics

        logger.warning(
            "Primary temporal reasoner failed status=%s; trying explicit fallback=%s",
            result.status,
            self.fallback_name,
        )
        fb_result, fb_diag = self.fallback.reason(
            temporal,
            baseline_overall=baseline_overall,
        )
        # Preserve primary failure metadata under details without secrets.
        details = dict(fb_result.details or {})
        details["primary_provider_status"] = result.status
        details["fallback"] = self.fallback_name
        fb_result = fb_result.model_copy(update={"details": details})
        if fb_diag.generation_kwargs is None:
            fb_diag.generation_kwargs = {}
        fb_diag.generation_kwargs = {
            **dict(fb_diag.generation_kwargs),
            "fallback_from": getattr(diagnostics, "provider", None) or "primary",
        }
        return fb_result, fb_diag


def create_temporal_reasoner(
    config: TemporalReasonerConfig = DEFAULT_TEMPORAL_REASONER,
) -> TemporalReasonerProvider:
    """Create the configured provider. Default demo provider is OpenRouter."""
    provider = (config.provider or "openrouter").strip().lower()
    fallback_name = (config.fallback or "none").strip().lower()

    if provider == "qwen_local_or_zerogpu":
        qwen_cfg = replace(
            config,
            model_id=config.qwen_model_id or config.model_id,
            provider="qwen_local_or_zerogpu",
            fallback="none",
        )
        return TemporalContextReasoner(qwen_cfg)

    # openrouter (demo default)
    primary = OpenRouterTemporalReasoner(config)
    if fallback_name == "qwen":
        qwen_cfg = replace(
            config,
            model_id=config.qwen_model_id,
            provider="qwen_local_or_zerogpu",
            fallback="none",
            enabled=True,
        )
        fallback = TemporalContextReasoner(qwen_cfg)
        return FallbackTemporalReasoner(
            primary,
            fallback=fallback,
            fallback_name="qwen",
        )
    return primary
