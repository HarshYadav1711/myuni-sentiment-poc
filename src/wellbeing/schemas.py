"""Structured results for the independent wellbeing / context classifier.

Package-local schemas keep this Phase 4 foundation isolated from the
pipeline ``src.schemas`` types (FinalTemporalAssessment, wellbeing gate, etc.)
until a later wiring pass. Same pattern as ``src.temporal.benchmark.schemas``.

No wellbeing score (0–10 / 0–100) is defined — model probabilities are
model evidence only, not clinical probabilities.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.wellbeing.labels import (
    RELEVANCE_LABELS,
    SIGNAL_LABELS,
    TARGET_LABELS,
)

ClassifierStatus = Literal[
    "ok",
    "insufficient_text",
    "classifier_unavailable",
    "error",
]

RelevanceLabel = Literal[
    "personal_wellbeing",
    "wellbeing_topic_only",
    "not_wellbeing_related",
    "ambiguous",
]

TargetLabel = Literal[
    "self",
    "other_person",
    "group_or_community",
    "institution_or_event",
    "general_or_unknown",
]

SignalLabel = Literal[
    "stress_or_overwhelm",
    "anxiety_or_fear_language",
    "loneliness_or_isolation",
    "hopelessness_like_language",
    "exhaustion_or_burnout_like_language",
    "self_directed_negativity",
    "interpersonal_distress",
    "academic_pressure",
    "positive_wellbeing_or_recovery",
]

SourceType = Literal["text", "speech_transcript", "ocr", "caption", "other"]


class ExclusiveClassification(BaseModel):
    """One exclusive (single-label) classification with full score vector."""

    model_config = ConfigDict(extra="forbid")

    label: Optional[str] = None
    scores: dict[str, float] = Field(default_factory=dict)
    confidence: Optional[float] = Field(
        default=None,
        description="Top-1 score (model evidence, not a clinical probability).",
    )


class SignalScore(BaseModel):
    """One multi-label signal with score and provisional selection flag."""

    model_config = ConfigDict(extra="forbid")

    signal: str
    score: float
    selected: bool = False


class ClassifierInputMetadata(BaseModel):
    """Non-sensitive metadata about the classified input."""

    model_config = ConfigDict(extra="forbid")

    source_type: SourceType = "text"
    input_character_count: int = 0
    truncated: bool = False


class ClassifierUncertainty(BaseModel):
    """Top-1 vs top-2 margins for exclusive heads (model evidence only)."""

    model_config = ConfigDict(extra="forbid")

    relevance_top1_top2_margin: Optional[float] = None
    target_top1_top2_margin: Optional[float] = None


class WellbeingClassificationResult(BaseModel):
    """Independent content-level wellbeing relevance / target / signals result.

    Does NOT invent a wellbeing score. Does NOT diagnose mental-health
    conditions. Missing or unusable text yields ``insufficient_text`` —
    missing is not neutral.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str
    status: ClassifierStatus
    relevance: ExclusiveClassification = Field(
        default_factory=ExclusiveClassification,
    )
    target: ExclusiveClassification = Field(
        default_factory=ExclusiveClassification,
    )
    signals: list[SignalScore] = Field(default_factory=list)
    input_metadata: ClassifierInputMetadata = Field(
        default_factory=ClassifierInputMetadata,
    )
    uncertainty: ClassifierUncertainty = Field(
        default_factory=ClassifierUncertainty,
    )
    error_code: Optional[str] = Field(
        default=None,
        description="Machine-readable failure code; never contains user text.",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Safe diagnostic summary; must not echo raw user text.",
    )


# Re-export taxonomy sizes for schema consumers / tests.
EXPECTED_RELEVANCE_LABELS = RELEVANCE_LABELS
EXPECTED_TARGET_LABELS = TARGET_LABELS
EXPECTED_SIGNAL_LABELS = SIGNAL_LABELS
