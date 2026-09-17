"""Structured results for the independent wellbeing / context classifier.

Phase 4A.5:
- dual independent attribution evidence (direct_self + reported_other)
- final attribution may be policy-derived ``unclear``
- legacy binary attribution retained for evaluation comparison only
- A–K and previous FH40 are regression/development (contaminated)
- fresh evaluation uses final_holdout_v2.py
"""

from __future__ import annotations

from typing import Any, Literal, Optional

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


# ---------------------------------------------------------------------------
# Phase 4B — shadow-mode pipeline evidence (non-authoritative)
# ---------------------------------------------------------------------------

WellbeingShadowStatus = Literal[
    "ok",
    "partial",
    "insufficient_text",
    "classifier_unavailable",
    "error",
]

ShadowSourceRole = Literal[
    "primary_text",
    "caption",
    "transcript",
    "speech_window",
]


class WellbeingShadowSourceResult(BaseModel):
    """One classified textual source in shadow mode. No raw text stored."""

    model_config = ConfigDict(extra="forbid")

    source_role: ShadowSourceRole
    classification: Optional[WellbeingClassificationResult] = None
    input_character_count: int = 0
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Non-text provenance (ids/indexes only).",
    )


class WellbeingShadowWindowResult(BaseModel):
    """Per-window shadow classification. No speech text stored."""

    model_config = ConfigDict(extra="forbid")

    window_index: int
    start: float
    end: float
    classification: Optional[WellbeingClassificationResult] = None
    input_character_count: int = 0
    usable_window: bool = False


class WellbeingShadowSummary(BaseModel):
    """Descriptive evidence counts only — not a wellbeing score."""

    model_config = ConfigDict(extra="forbid")

    evaluated_source_count: int = 0
    eligible_source_count: int = 0
    uncertain_source_count: int = 0
    not_eligible_source_count: int = 0
    selected_signal_counts: dict[str, int] = Field(default_factory=dict)
    evaluated_window_count: int = 0
    eligible_window_count: int = 0
    uncertain_window_count: int = 0
    not_eligible_window_count: int = 0
    items_submitted: int = 0
    windows_submitted: int = 0


class WellbeingShadowAnalysis(BaseModel):
    """Parallel non-authoritative wellbeing classifier evidence (Phase 4B).

    Does not alter FinalTemporalAssessment, sentiment, fusion, or the
    existing temporal wellbeing gate.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["shadow"] = "shadow"
    status: WellbeingShadowStatus = "insufficient_text"
    model_id: Optional[str] = None
    source_results: list[WellbeingShadowSourceResult] = Field(default_factory=list)
    window_results: list[WellbeingShadowWindowResult] = Field(default_factory=list)
    summary: WellbeingShadowSummary = Field(default_factory=WellbeingShadowSummary)
    evidence_context: Optional["WellbeingEvidenceContext"] = Field(
        default=None,
        description=(
            "Phase 4C.1 deterministic evidence aggregation. Optional; "
            "does not alter FinalTemporalAssessment or client indicator."
        ),
    )
    policy_candidate: Optional["WellbeingPolicyCandidate"] = Field(
        default=None,
        description=(
            "Phase 4C.2 deterministic shadow policy candidate. Optional; "
            "does not alter FinalTemporalAssessment or client indicator."
        ),
    )
    authority_comparison: Optional["WellbeingAuthorityComparison"] = Field(
        default=None,
        description=(
            "Phase 4C.5 non-authoritative legacy-vs-candidate comparison. "
            "Does not alter FinalTemporalAssessment."
        ),
    )
    processing_seconds: Optional[float] = None
    affects_final_assessment: Literal[False] = False
    note: str = (
        "Shadow evidence only. Does not alter client-facing wellbeing "
        "indicator, sentiment, fusion, or temporal assessment."
    )
    error_code: Optional[str] = None


# ---------------------------------------------------------------------------
# Phase 4C.1 — deterministic wellbeing evidence aggregation (non-policy)
#
# Global transcript/source and local windows serve DIFFERENT roles:
# - GLOBAL: authorship / context / whole-content semantics
# - WINDOWS: temporal localization / recurrence / transitions
# They are not equivalent votes. Signal probabilities are never averaged.
# ---------------------------------------------------------------------------

WellbeingEvidenceStatus = Literal[
    "ok",
    "empty",
    "insufficient",
    "error",
]

WellbeingConflictDiagnostic = Literal[
    "global_eligible_no_local_support",
    "local_eligible_without_global_eligibility",
    "global_recovery_local_distress",
    "global_distress_local_recovery",
]


class WellbeingSignalTemporalEvidence(BaseModel):
    """Per-signal recurrence across eligible windows (descriptive only)."""

    model_config = ConfigDict(extra="forbid")

    signal: str
    window_count: int = 0
    window_indices: list[int] = Field(default_factory=list)
    first_start: Optional[float] = None
    last_end: Optional[float] = None


class WellbeingEligibleRun(BaseModel):
    """One consecutive run of eligible wellbeing windows."""

    model_config = ConfigDict(extra="forbid")

    start_window: int
    end_window: int
    start: float
    end: float
    window_count: int
    duration_seconds: float
    signal_ids: list[str] = Field(default_factory=list)


class WellbeingTemporalEvidence(BaseModel):
    """Aggregated temporal/window wellbeing evidence (descriptive only).

    ``eligible_window_fraction`` is descriptive recurrence, not a
    wellbeing probability or concern score.
    """

    model_config = ConfigDict(extra="forbid")

    evaluated_window_count: int = 0
    eligible_window_count: int = 0
    uncertain_window_count: int = 0
    not_eligible_window_count: int = 0

    eligible_window_indices: list[int] = Field(default_factory=list)
    eligible_window_fraction: Optional[float] = None

    first_eligible_start: Optional[float] = None
    last_eligible_end: Optional[float] = None

    eligible_runs: list[WellbeingEligibleRun] = Field(default_factory=list)
    longest_eligible_run_windows: int = 0
    longest_eligible_run_seconds: float = 0.0

    distress_window_count: int = 0
    recovery_window_count: int = 0
    mixed_signal_window_count: int = 0

    signal_evidence: list[WellbeingSignalTemporalEvidence] = Field(
        default_factory=list,
    )


class WellbeingGlobalEvidence(BaseModel):
    """Global (non-window) source evidence — typically full transcript.

    Preserves classification fields only. No averaged scores.
    """

    model_config = ConfigDict(extra="forbid")

    source_present: bool = False
    source_role: Optional[ShadowSourceRole] = None
    classification_status: Optional[ClassifierStatus] = None
    relevance: Optional[str] = None
    target: Optional[str] = None
    eligibility_status: Optional[EligibilityStatus] = None
    personal_wellbeing_eligible: bool = False
    final_attribution: Optional[str] = None
    selected_signals: list[str] = Field(default_factory=list)


class WellbeingEvidenceContext(BaseModel):
    """Deterministic wellbeing evidence feature layer (Phase 4C.1).

    Not a concern policy. No mental-health score. No final concern category.
    ``affects_final_assessment`` is always False in this phase.
    """

    model_config = ConfigDict(extra="forbid")

    status: WellbeingEvidenceStatus = "empty"
    global_evidence: WellbeingGlobalEvidence = Field(
        default_factory=WellbeingGlobalEvidence,
    )
    temporal_evidence: WellbeingTemporalEvidence = Field(
        default_factory=WellbeingTemporalEvidence,
    )
    evidence_ids: list[str] = Field(default_factory=list)
    conflict_diagnostics: list[WellbeingConflictDiagnostic] = Field(
        default_factory=list,
        description="Machine-readable global/local disagreement codes only.",
    )
    affects_final_assessment: Literal[False] = False
    note: str = (
        "Wellbeing evidence aggregation only. Global and window evidence "
        "serve different roles and are not majority-voted. No concern "
        "category or wellbeing score is produced in Phase 4C.1."
    )


# ---------------------------------------------------------------------------
# Phase 4C.2 — deterministic shadow wellbeing policy candidate (non-authoritative)
#
# Interprets WellbeingEvidenceContext only. Shadow-only: never replaces
# compute_wellbeing_indicator() / FinalTemporalAssessment / Gradio output.
# Categories are content-level evidence labels — not diagnoses or clinical risk.
# ---------------------------------------------------------------------------

WellbeingPolicyStatus = Literal[
    "ok",
    "insufficient_evidence",
    "conflict",
    "unavailable",
]

WellbeingPolicyIndicator = Literal[
    "low_concern",
    "moderate_concern",
    "high_concern",
    "insufficient_evidence",
]

WellbeingLocalSupportLevel = Literal[
    "none",
    "isolated",
    "recurrent",
    "persistent",
]

WellbeingDistressPattern = Literal[
    "none",
    "isolated",
    "recurrent",
    "persistent",
    "mixed_with_recovery",
]

WellbeingRecoveryPattern = Literal[
    "none",
    "global_only",
    "local_only",
    "recurrent",
    "mixed_with_distress",
]


class WellbeingPolicyCandidate(BaseModel):
    """Experimental deterministic concern candidate (Phase 4C.2).

    Shadow-only interpretation of ``WellbeingEvidenceContext``.
    Not a clinical judgment, triage score, or client-facing indicator.
    """

    model_config = ConfigDict(extra="forbid")

    status: WellbeingPolicyStatus = "insufficient_evidence"
    indicator: WellbeingPolicyIndicator = "insufficient_evidence"
    reason_codes: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    conflict_evidence_ids: list[str] = Field(default_factory=list)
    global_eligible: bool = False
    local_support_level: WellbeingLocalSupportLevel = "none"
    distress_pattern: WellbeingDistressPattern = "none"
    recovery_pattern: WellbeingRecoveryPattern = "none"
    policy_version: Literal["phase4c2-v1"] = "phase4c2-v1"
    affects_final_assessment: Literal[False] = False
    note: str = (
        "Shadow policy candidate only. Content-level wellbeing evidence "
        "interpretation — not a diagnosis, clinical risk level, or "
        "replacement for the existing temporal wellbeing indicator."
    )


# ---------------------------------------------------------------------------
# Phase 4C.5 — authority migration comparison (control plane; non-authoritative)
#
# VIDEO-first migration design. Default mode is legacy. Candidate authority
# is NOT activated in this phase. Comparison never mutates either result.
# ---------------------------------------------------------------------------

WellbeingAuthorityMode = Literal["legacy", "compare", "candidate"]

WellbeingAuthorityAgreement = Literal[
    "same",
    "different",
    "candidate_unavailable",
    "not_compared",
]

WellbeingCandidateOutcomeKind = Literal[
    "available",
    "semantic_abstention",
    "technical_failure",
    "unavailable",
]


class WellbeingAuthorityComparison(BaseModel):
    """Non-authoritative legacy vs candidate comparison diagnostics.

    Attached to shadow analysis only. Never alters FinalTemporalAssessment
    in Phase 4C.5. ``affects_final_assessment`` is always False here.
    """

    model_config = ConfigDict(extra="forbid")

    mode: WellbeingAuthorityMode = "legacy"
    migration_scope: Literal["video"] = "video"
    legacy_indicator: Optional[str] = None
    candidate_indicator: Optional[str] = None
    candidate_status: Optional[str] = None
    agreement: WellbeingAuthorityAgreement = "not_compared"
    legacy_available: bool = False
    candidate_available: bool = False
    candidate_outcome_kind: WellbeingCandidateOutcomeKind = "unavailable"
    fallback_required: bool = False
    fallback_reason: Optional[str] = None
    comparison_reason_codes: list[str] = Field(default_factory=list)
    candidate_policy_version: Optional[str] = None
    candidate_evidence_ids: list[str] = Field(default_factory=list)
    affects_final_assessment: Literal[False] = False
    note: str = (
        "Authority migration comparison only. Legacy remains authoritative "
        "while mode is legacy/compare. Candidate mode is not activated in "
        "Phase 4C.5. Disagreement does not imply either side is correct."
    )


# ---------------------------------------------------------------------------
# Phase 4C.7 — authority activation readiness design (NOT activated)
#
# Typed readiness + audit contracts for a future candidate-authority path.
# Evaluating readiness must NEVER route FinalTemporalAssessment.
# ---------------------------------------------------------------------------

WellbeingAuthoritySource = Literal[
    "legacy",
    "candidate",
    "legacy_technical_fallback",
    "blocked",
]

WellbeingAuthorityScope = Literal["video", "text", "image", "audio"]


class WellbeingAuthorityReadiness(BaseModel):
    """Explicit multi-flag readiness for candidate authority activation.

    Do not infer readiness from a single boolean config flag.
    ``ready_for_activation`` is True only when every required prerequisite
    is True. Phase 4C.7 ships design/evaluation helpers only — activation
    remains blocked.
    """

    model_config = ConfigDict(extra="forbid")

    readiness_version: str = "phase4c7-v1"
    candidate_policy_validated: bool = False
    shadow_replay_validated: bool = False
    live_compare_validated: bool = False
    production_like_compare_validated: bool = False
    runtime_acceptable: bool = False
    technical_fallback_validated: bool = False
    rollback_validated: bool = False
    ui_semantics_approved: bool = False
    observability_ready: bool = False
    ready_for_activation: bool = False
    blocking_reasons: list[str] = Field(default_factory=list)
    note: str = (
        "Authority readiness design only. Candidate authority is not "
        "activated. Phase 4C.6 compare with reasoner disabled does not "
        "satisfy production_like_compare_validated."
    )


class WellbeingAuthorityDecision(BaseModel):
    """Safe diagnostic audit record for future authoritative runs.

    Design schema only in Phase 4C.7. No raw text. Never authorizes FTA
    while ``affects_final_assessment`` remains False.
    """

    model_config = ConfigDict(extra="forbid")

    requested_mode: WellbeingAuthorityMode = "legacy"
    resolved_mode: WellbeingAuthorityMode = "legacy"
    authority_source: WellbeingAuthoritySource = "legacy"
    migration_scope: WellbeingAuthorityScope = "video"
    candidate_policy_version: Optional[str] = None
    candidate_status: Optional[str] = None
    candidate_indicator: Optional[str] = None
    legacy_indicator_if_available: Optional[str] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    candidate_failure_kind: Optional[WellbeingCandidateOutcomeKind] = None
    legacy_used_as_technical_fallback: bool = False
    readiness_version: Optional[str] = None
    evidence_ids: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    affects_final_assessment: Literal[False] = False
    note: str = (
        "Authority decision audit design only. Does not alter "
        "FinalTemporalAssessment in Phase 4C.7."
    )


class WellbeingPolicyStabilitySnapshot(BaseModel):
    """Policy-level stability fields across repeated controlled runs.

    Window-index movement alone is not treated as policy instability.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = ""
    candidate_indicator: Optional[str] = None
    candidate_status: Optional[str] = None
    local_support_level: Optional[str] = None
    distress_pattern: Optional[str] = None
    recovery_pattern: Optional[str] = None
    eligible_window_indices: list[int] = Field(default_factory=list)
    policy_version: Optional[str] = None


class WellbeingAuthorityObservabilityCounters(BaseModel):
    """Required counter names before activation (values are runtime tallies).

    No user text, secrets, or clinical interpretation.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_available_count: int = 0
    candidate_semantic_abstention_count: int = 0
    candidate_technical_failure_count: int = 0
    candidate_low_count: int = 0
    candidate_moderate_count: int = 0
    candidate_high_count: int = 0
    candidate_insufficient_count: int = 0
    candidate_legacy_disagreement_count: int = 0
    technical_fallback_count: int = 0
    processing_seconds: float = 0.0


class WellbeingIndicatorDisplayMapping(BaseModel):
    """Future UI terminology mapping (design only; not wired to Gradio).

    Product must approve before replacing legacy \"Stress\" display labels.
    """

    model_config = ConfigDict(extra="forbid")

    product_label: str = "Wellbeing Indicator"
    low_concern: str = "Low Concern"
    moderate_concern: str = "Moderate Concern"
    high_concern: str = "High Concern"
    insufficient_evidence: str = "Insufficient Evidence"
    approved: bool = False
    note: str = (
        "Do not map to Low/Moderate/High Stress without explicit product "
        "approval. Internal enum values remain *_concern."
    )


EXPECTED_RELEVANCE_LABELS = RELEVANCE_LABELS
EXPECTED_TARGET_LABELS = TARGET_LABELS
EXPECTED_SIGNAL_LABELS = SIGNAL_LABELS
EXPECTED_RAW_ATTRIBUTION_LABELS = RAW_ATTRIBUTION_LABELS
EXPECTED_FINAL_ATTRIBUTION_LABELS = FINAL_ATTRIBUTION_LABELS
