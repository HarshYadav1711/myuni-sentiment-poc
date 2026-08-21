"""Pydantic schemas for MyUni sentiment analysis POC outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


SentimentLabel = Literal["positive", "neutral", "negative"]
ActivityType = Literal["text", "image", "video"]


class SentimentEvidence(BaseModel):
    """Single-modality (or overall) sentiment evidence."""

    label: SentimentLabel
    score: float = Field(
        ...,
        description="POC sentiment score approximately in [-1, +1].",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Optional[dict[str, float]] = None
    model: Optional[str] = None
    details: Optional[dict[str, Any]] = None


class ModalityBundle(BaseModel):
    """Per-modality evidence. Unused modalities are omitted."""

    text: Optional[SentimentEvidence] = None


class AnalysisBlock(BaseModel):
    overall: SentimentEvidence
    modalities: ModalityBundle


class InputMetadata(BaseModel):
    text_length: Optional[int] = None
    text_preview: Optional[str] = None


class ActivityAnalysisResult(BaseModel):
    """Standardized activity-level sentiment result."""

    activity_id: str
    user_id: Optional[str] = None
    activity_type: ActivityType
    input: InputMetadata
    analysis: AnalysisBlock
    processed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def model_dump_json_compatible(self) -> dict[str, Any]:
        """Serialize with ISO timestamps for CLI / file output."""
        return self.model_dump(mode="json")
