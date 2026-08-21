"""Central MyUni sentiment analysis pipeline."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from src.analyzers.text import TextSentimentAnalyzer
from src.schemas import (
    ActivityAnalysisResult,
    ActivityInput,
    AnalysisBlock,
    InputMetadata,
    ModalityBundle,
)

logger = logging.getLogger(__name__)

_PREVIEW_LEN = 120


class MyUniSentimentPipeline:
    """Routes activities through modality analyzers and returns standardized results.

    Milestone 2: text activities are fully analyzed; image/video are rejected here
    as unsupported (batch layer reports them without fake sentiment).
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
        """Analyze a free-form text activity (CLI convenience path)."""
        cleaned = self._text_analyzer.validate_text(text)
        resolved_activity_id = activity_id or f"ACT-{uuid.uuid4().hex[:8].upper()}"
        return self._analyze_text_content(
            cleaned,
            activity_id=resolved_activity_id,
            user_id=user_id,
            created_at=None,
            content_kind=None,
            extra=None,
            media_path=None,
        )

    def analyze_activity(self, activity: ActivityInput) -> ActivityAnalysisResult:
        """Analyze a validated ActivityInput.

        Only ``text`` activities are implemented in Milestone 2.
        """
        if activity.activity_type != "text":
            raise NotImplementedError(
                f"activity_type={activity.activity_type!r} is not implemented yet",
            )

        assert activity.text is not None  # enforced by ActivityInput
        cleaned = self._text_analyzer.validate_text(activity.text)
        return self._analyze_text_content(
            cleaned,
            activity_id=activity.activity_id,
            user_id=activity.user_id,
            created_at=activity.created_at,
            content_kind=activity.content_kind,
            extra=activity.metadata,
            media_path=activity.media_path,
        )

    def _analyze_text_content(
        self,
        cleaned: str,
        *,
        activity_id: str,
        user_id: Optional[str],
        created_at: object,
        content_kind: object,
        extra: object,
        media_path: Optional[str],
    ) -> ActivityAnalysisResult:
        evidence = self._text_analyzer.analyze(cleaned)
        preview = cleaned if len(cleaned) <= _PREVIEW_LEN else f"{cleaned[:_PREVIEW_LEN]}..."

        logger.info(
            "Analyzed text activity_id=%s user_id=%s label=%s score=%.3f",
            activity_id,
            user_id,
            evidence.label,
            evidence.score,
        )

        return ActivityAnalysisResult(
            activity_id=activity_id,
            user_id=user_id,
            activity_type="text",
            input=InputMetadata(
                text_length=len(cleaned),
                text_preview=preview,
                media_path=media_path,
                created_at=created_at,  # type: ignore[arg-type]
                content_kind=content_kind,  # type: ignore[arg-type]
                extra=extra,  # type: ignore[arg-type]
            ),
            analysis=AnalysisBlock(
                overall=evidence.model_copy(
                    update={"details": {"source_modality": "text"}},
                ),
                modalities=ModalityBundle(text=evidence),
            ),
        )
