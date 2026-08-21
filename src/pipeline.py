"""Central MyUni sentiment analysis pipeline."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from src.analyzers.text import TextSentimentAnalyzer
from src.schemas import (
    ActivityAnalysisResult,
    AnalysisBlock,
    InputMetadata,
    ModalityBundle,
)

logger = logging.getLogger(__name__)

_PREVIEW_LEN = 120


class MyUniSentimentPipeline:
    """Routes activities through modality analyzers and returns standardized results.

    Milestone 1 supports text activities only.
    """

    def __init__(self, text_analyzer: Optional[TextSentimentAnalyzer] = None) -> None:
        self._text_analyzer = text_analyzer or TextSentimentAnalyzer()

    @property
    def text_analyzer(self) -> TextSentimentAnalyzer:
        return self._text_analyzer

    def analyze_text(
        self,
        text: object,
        *,
        user_id: Optional[str] = None,
        activity_id: Optional[str] = None,
    ) -> ActivityAnalysisResult:
        """Analyze a text activity end-to-end."""
        cleaned = self._text_analyzer.validate_text(text)
        evidence = self._text_analyzer.analyze(cleaned)

        resolved_activity_id = activity_id or f"ACT-{uuid.uuid4().hex[:8].upper()}"
        preview = cleaned if len(cleaned) <= _PREVIEW_LEN else f"{cleaned[:_PREVIEW_LEN]}..."

        logger.info(
            "Analyzed text activity_id=%s user_id=%s label=%s score=%.3f",
            resolved_activity_id,
            user_id,
            evidence.label,
            evidence.score,
        )

        return ActivityAnalysisResult(
            activity_id=resolved_activity_id,
            user_id=user_id,
            activity_type="text",
            input=InputMetadata(
                text_length=len(cleaned),
                text_preview=preview,
            ),
            analysis=AnalysisBlock(
                overall=evidence.model_copy(
                    update={"details": {"source_modality": "text"}},
                ),
                modalities=ModalityBundle(text=evidence),
            ),
        )
