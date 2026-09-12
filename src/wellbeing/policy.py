"""Conservative personal-wellbeing eligibility policy (Phase 4A.5).

Dual independent attribution evidence drives production:

- direct_self_score >= DIRECT_SELF_MIN_SCORE
- reported_other_score < REPORTED_OTHER_BLOCK_SCORE
  → self_experience (eligible path continues)

- reported_other_score >= REPORTED_OTHER_BLOCK_SCORE
  and direct_self clearly insufficient
  → not_self_experience

- both strong / both weak / otherwise
  → unclear (uncertain eligibility)

Legacy binary attribution is comparison-only and must not affect decisions.

Mixed self+other: if both heads are strong, prefer unclear rather than
auto-blocking solely because other-person evidence is present — unless
reported_other meets the block threshold AND direct_self is insufficient.

Development/calibration thresholds are POC engineering values, not clinical
cutoffs. A–K, calibration, attribution_dev, and previous FH40 are
regression/development (contaminated). Fresh holdout is final_holdout_v2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.wellbeing.schemas import (
    AttributionEvidence,
    EligibilityStatus,
    SelfAttributionResult,
)


@dataclass(frozen=True)
class EligibilityDecision:
    """Policy outcome for one classification."""

    status: EligibilityStatus
    reasons: list[str]
    finalized_attribution: SelfAttributionResult

    @property
    def personal_wellbeing_eligible(self) -> bool:
        return self.status == "eligible"


def _evidence_from_attribution(
    attribution: SelfAttributionResult,
) -> AttributionEvidence:
    if attribution.attribution_evidence.evidence_status != "not_evaluated":
        return attribution.attribution_evidence
    scores = attribution.scores or {}
    if "direct_self_experience" in scores or "reported_other_experience" in scores:
        return AttributionEvidence(
            direct_self_score=float(scores.get("direct_self_experience", 0.0)),
            reported_other_score=float(
                scores.get("reported_other_experience", 0.0),
            ),
            evidence_status="ok" if attribution.status == "ok" else (
                attribution.status  # type: ignore[arg-type]
            ),
            error_code=attribution.error_code,
        )
    return attribution.attribution_evidence


def finalize_self_attribution(
    attribution: SelfAttributionResult,
    *,
    direct_self_min_score: float,
    reported_other_block_score: float,
) -> SelfAttributionResult:
    """Derive final attribution label from dual-head evidence.

    Does not rewrite raw evidence scores. Legacy binary is preserved
    untouched and ignored for the decision.
    """
    evidence = _evidence_from_attribution(attribution)
    legacy = attribution.legacy_binary

    if attribution.status != "ok":
        return SelfAttributionResult(
            status=attribution.status,
            raw_label=attribution.raw_label,
            final_label=None,
            label=None,
            scores=dict(attribution.scores),
            top_score=attribution.top_score,
            top1_top2_margin=attribution.top1_top2_margin,
            attribution_evidence=evidence,
            legacy_binary=legacy,
            error_code=attribution.error_code,
        )

    direct = evidence.direct_self_score
    other = evidence.reported_other_score
    if direct is None or other is None:
        finalized_evidence = AttributionEvidence(
            direct_self_score=direct,
            reported_other_score=other,
            evidence_status="insufficient",
            error_code=evidence.error_code,
        )
        return SelfAttributionResult(
            status="ok",
            raw_label=None,
            final_label="unclear",
            label="unclear",
            scores=dict(attribution.scores),
            top_score=attribution.top_score,
            top1_top2_margin=attribution.top1_top2_margin,
            attribution_evidence=finalized_evidence,
            legacy_binary=legacy,
            error_code=None,
        )

    direct_f = float(direct)
    other_f = float(other)
    self_min = float(direct_self_min_score)
    other_block = float(reported_other_block_score)

    if direct_f >= self_min and other_f < other_block:
        final = "self_experience"
        evidence_status = "ok"
    elif other_f >= other_block and direct_f < self_min:
        # Reported/quoted other dominates and self evidence is insufficient.
        final = "not_self_experience"
        evidence_status = "ok"
    elif direct_f >= self_min and other_f >= other_block:
        # Strong mixed evidence — prefer abstention over false personal claim.
        final = "unclear"
        evidence_status = "conflict"
    else:
        final = "unclear"
        evidence_status = "insufficient"

    finalized_evidence = AttributionEvidence(
        direct_self_score=direct_f,
        reported_other_score=other_f,
        evidence_status=evidence_status,
        error_code=None,
    )
    return SelfAttributionResult(
        status="ok",
        raw_label=final if final != "unclear" else None,
        final_label=final,
        label=final,
        scores={
            "direct_self_experience": direct_f,
            "reported_other_experience": other_f,
        },
        top_score=max(direct_f, other_f),
        top1_top2_margin=abs(direct_f - other_f),
        attribution_evidence=finalized_evidence,
        legacy_binary=legacy,
        error_code=None,
    )


def decide_eligibility(
    *,
    classifier_status: str,
    relevance_label: Optional[str],
    target_label: Optional[str],
    relevance_margin: Optional[float],
    target_margin: Optional[float],
    attribution: SelfAttributionResult,
    relevance_min_margin: float,
    target_min_margin: float,
    direct_self_min_score: float,
    reported_other_block_score: float,
    # Legacy kwargs accepted but ignored for production dual-head path.
    attribution_min_margin: float = 0.0,
    attribution_min_top_score: float = 0.0,
) -> EligibilityDecision:
    """Apply dual-head uncertainty-aware personal eligibility policy."""
    _ = attribution_min_margin, attribution_min_top_score  # unused (compat)
    empty_attr = SelfAttributionResult(
        status=attribution.status,
        attribution_evidence=attribution.attribution_evidence,
        legacy_binary=attribution.legacy_binary,
    )

    if classifier_status == "insufficient_text":
        return EligibilityDecision(
            "not_eligible",
            ["insufficient_text"],
            empty_attr,
        )
    if classifier_status == "classifier_unavailable":
        return EligibilityDecision(
            "not_eligible",
            ["classifier_unavailable"],
            empty_attr,
        )
    if classifier_status == "error":
        return EligibilityDecision(
            "not_eligible",
            ["classifier_error"],
            empty_attr,
        )
    if classifier_status != "ok":
        return EligibilityDecision("not_eligible", ["status_not_ok"], empty_attr)

    reasons: list[str] = []
    if relevance_label != "personal_wellbeing":
        reasons.append("relevance_not_personal")
    if target_label != "self":
        reasons.append("target_not_self")
    if reasons:
        return EligibilityDecision("not_eligible", reasons, empty_attr)

    if attribution.status == "error":
        return EligibilityDecision(
            "not_eligible",
            ["classifier_error"],
            attribution,
        )
    if attribution.status == "not_evaluated":
        return EligibilityDecision(
            "not_eligible",
            ["self_attribution_not_evaluated"],
            attribution,
        )
    if attribution.status != "ok":
        return EligibilityDecision(
            "not_eligible",
            ["classifier_error"],
            attribution,
        )

    finalized = finalize_self_attribution(
        attribution,
        direct_self_min_score=direct_self_min_score,
        reported_other_block_score=reported_other_block_score,
    )

    if finalized.final_label == "not_self_experience":
        return EligibilityDecision(
            "not_eligible",
            ["self_attribution_not_self", "reported_other_score_blocks"],
            finalized,
        )
    if finalized.final_label == "unclear" or finalized.final_label is None:
        unclear_reasons = ["self_attribution_uncertain"]
        ev = finalized.attribution_evidence
        if ev.evidence_status == "conflict":
            unclear_reasons.append("attribution_evidence_conflict")
        elif ev.evidence_status == "insufficient":
            unclear_reasons.append("attribution_evidence_insufficient")
        direct = ev.direct_self_score
        other = ev.reported_other_score
        if direct is not None and float(direct) < float(direct_self_min_score):
            unclear_reasons.append("direct_self_score_too_low")
        if other is not None and float(other) >= float(
            reported_other_block_score,
        ):
            unclear_reasons.append("reported_other_score_blocks")
        return EligibilityDecision("uncertain", unclear_reasons, finalized)
    if finalized.final_label != "self_experience":
        return EligibilityDecision(
            "uncertain",
            ["self_attribution_uncertain"],
            finalized,
        )

    margin_reasons: list[str] = []
    if relevance_margin is None or float(relevance_margin) < float(
        relevance_min_margin,
    ):
        margin_reasons.append("relevance_margin_too_small")
    if target_margin is None or float(target_margin) < float(target_min_margin):
        margin_reasons.append("target_margin_too_small")
    if margin_reasons:
        return EligibilityDecision("uncertain", margin_reasons, finalized)

    return EligibilityDecision("eligible", [], finalized)
