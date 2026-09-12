"""Structured results for the independent wellbeing / context classifier.

Phase 4A.5:
- dual independent attribution evidence (direct_self + reported_other)
- final attribution may be policy-derived ``unclear``
- legacy binary attribution retained for evaluation comparison only
- A–K and previous FH40 are regression/development (contaminated)
- fresh evaluation uses final_holdout_v2.py
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.wellbeing.labels import (
    FINAL_ATTRIBUTION_LABELS,
    RAW_ATTRIBUTION_LABELS,
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

RawSelfAttributionLabel = Literal[
    "self_experience",
    "not_self_experience",
]

FinalSelfAttributionLabel = Literal[
    "self_experience",
    "not_self_experience",
    "unclear",
]

AttributionEvidenceStatus = Literal[
    "ok",
    "not_evaluated",
    "error",
    "insufficient",
    "conflict",
]

AttributionStatus = Literal["ok", "not_evaluated", "error"]

EligibilityStatus = Literal["eligible", "not_eligible", "uncertain"]

SourceType = Literal["text", "speech_transcript", "ocr", "caption", "other"]

EligibilityReason = Literal[
    "status_not_ok",
    "insufficient_text",
    "classifier_unavailable",
    "classifier_error",
    "relevance_not_personal",
    "target_not_self",
    "self_attribution_not_self",
    "self_attribution_uncertain",
    "self_attribution_not_evaluated",
    "relevance_margin_too_small",
    "target_margin_too_small",
    "direct_self_score_too_low",
    "reported_other_score_blocks",
    "attribution_evidence_conflict",
    "attribution_evidence_insufficient",
]


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
    """One multi-label signal: raw score vs provisional threshold vs selection."""

    model_config = ConfigDict(extra="forbid")

    signal: str
    score: float
    threshold_passed: bool = False
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
    attribution_top1_top2_margin: Optional[float] = None


class AttributionEvidence(BaseModel):
    """Independent dual-head attribution evidence (Phase 4A.5).

    Scores are raw multi_label NLI evidence. No user text is stored.
    """

    model_config = ConfigDict(extra="forbid")

    direct_self_score: Optional[float] = None
    reported_other_score: Optional[float] = None
    evidence_status: AttributionEvidenceStatus = "not_evaluated"
    error_code: Optional[str] = None


class LegacyBinaryAttribution(BaseModel):
    """Legacy exclusive binary attribution (evaluation comparison only).

    Never drives production eligibility when dual-head policy is active.
    """

    model_config = ConfigDict(extra="forbid")

    status: AttributionStatus = "not_evaluated"
    raw_label: Optional[str] = None
    scores: dict[str, float] = Field(default_factory=dict)
    top_score: Optional[float] = None
    top1_top2_margin: Optional[float] = None
    error_code: Optional[str] = None


class SelfAttributionResult(BaseModel):
    """Conditional self-attribution guard (dual-head primary).

    Production decision uses ``attribution_evidence`` + policy thresholds.
    ``legacy_binary`` is comparison-only and must not affect eligibility.
    """

    model_config = ConfigDict(extra="forbid")

    status: AttributionStatus = "not_evaluated"
    # Compat: raw_label may mirror policy outcome source or legacy winner.
    raw_label: Optional[str] = Field(
        default=None,
        description=(
            "For dual-head: policy precursor summary label when available; "
            "legacy binary winner when comparison mode stores binary scores."
        ),
    )
    final_label: Optional[str] = Field(
        default=None,
        description=(
            "Policy outcome: self_experience | not_self_experience | unclear."
        ),
    )
    # Compat: ``label`` mirrors final_label for eligibility consumers.
    label: Optional[str] = Field(
        default=None,
        description="Alias of final_label (policy outcome).",
    )
    scores: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Dual-head raw scores keyed by direct_self_experience / "
            "reported_other_experience (production). Legacy binary keys may "
            "appear only inside legacy_binary."
        ),
    )
    top_score: Optional[float] = None
    top1_top2_margin: Optional[float] = None
    attribution_evidence: AttributionEvidence = Field(
        default_factory=AttributionEvidence,
    )
    legacy_binary: Optional[LegacyBinaryAttribution] = Field(
        default=None,
        description="Evaluation-only binary comparison; ignored by policy.",
    )
    error_code: Optional[str] = None


class WellbeingClassificationResult(BaseModel):
    """Independent content-level wellbeing classification result."""

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
    self_attribution: SelfAttributionResult = Field(
        default_factory=SelfAttributionResult,
    )
    input_metadata: ClassifierInputMetadata = Field(
        default_factory=ClassifierInputMetadata,
    )
    uncertainty: ClassifierUncertainty = Field(
        default_factory=ClassifierUncertainty,
    )
    eligibility_status: EligibilityStatus = Field(
        default="not_eligible",
        description=(
            "POC policy outcome: eligible / not_eligible / uncertain. "
            "Not a clinical judgment."
        ),
    )
    personal_wellbeing_eligible: bool = Field(
        default=False,
        description="True iff eligibility_status == 'eligible' (compat flag).",
    )
    eligibility_reasons: list[str] = Field(
        default_factory=list,
        description="Machine-readable policy reasons; never user text.",
    )
    error_code: Optional[str] = Field(default=None)
    error_message: Optional[str] = Field(default=None)


EXPECTED_RELEVANCE_LABELS = RELEVANCE_LABELS
EXPECTED_TARGET_LABELS = TARGET_LABELS
EXPECTED_SIGNAL_LABELS = SIGNAL_LABELS
EXPECTED_RAW_ATTRIBUTION_LABELS = RAW_ATTRIBUTION_LABELS
EXPECTED_FINAL_ATTRIBUTION_LABELS = FINAL_ATTRIBUTION_LABELS
