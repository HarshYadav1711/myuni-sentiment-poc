"""Phase 4C.5 — wellbeing authority migration comparison control plane.

Pure deterministic logic. No models, network, transcripts, or sentiment.

VIDEO-first scope. Default authority remains the legacy temporal gate.
Candidate mode is reserved and not activated in this phase.

Technical failure ≠ semantic abstention:
  technical → classifier/evidence/policy unavailable/error
  semantic  → insufficient_evidence / conflict as successful conservative outcome
"""

from __future__ import annotations

from typing import Optional, Sequence

from src.config import (
    resolve_wellbeing_authority_mode,
    resolve_wellbeing_candidate_authority_activated,
)
from src.wellbeing.schemas import (
    WellbeingAuthorityAgreement,
    WellbeingAuthorityComparison,
    WellbeingAuthorityMode,
    WellbeingCandidateOutcomeKind,
    WellbeingPolicyCandidate,
)

# Short forms for disagreement taxonomy codes.
_INDICATOR_SHORT = {
    "low_concern": "low",
    "moderate_concern": "moderate",
    "high_concern": "high",
    "insufficient_evidence": "insufficient",
}

TECHNICAL_FAILURE_REASON_CODES = frozenset(
    {
        "shadow_disabled",
        "classifier_unavailable",
        "classifier_error",
        "evidence_unavailable",
        "policy_unavailable",
        "unsupported_modality",
        "candidate_mode_not_activated",
        "global_unavailable",
    },
)

SEMANTIC_ABSTENTION_REASON_CODES = frozenset(
    {
        "global_not_eligible",
        "global_uncertain",
        "no_personal_wellbeing_evidence",
        "local_eligible_without_global_eligibility",
        "mixed_distress_recovery",
        "global_local_conflict",
        "candidate_conflict",
        "candidate_insufficient",
    },
)


def _short(indicator: Optional[str]) -> Optional[str]:
    if indicator is None:
        return None
    return _INDICATOR_SHORT.get(indicator, indicator)


def disagreement_reason_code(
    legacy_indicator: Optional[str],
    candidate_indicator: Optional[str],
) -> Optional[str]:
    """Deterministic disagreement taxonomy code (no quality judgment)."""
    if legacy_indicator is None or candidate_indicator is None:
        return None
    if legacy_indicator == candidate_indicator:
        return None
    left = _short(legacy_indicator)
    right = _short(candidate_indicator)
    if left is None or right is None:
        return None
    return f"legacy_{left}_candidate_{right}"


def classify_candidate_outcome(
    policy_candidate: Optional[WellbeingPolicyCandidate],
    *,
    shadow_enabled: bool = True,
    shadow_status: Optional[str] = None,
) -> WellbeingCandidateOutcomeKind:
    """Distinguish technical failure from semantic abstention / availability."""
    if not shadow_enabled:
        return "unavailable"
    if policy_candidate is None:
        if shadow_status in {"error", "classifier_unavailable"}:
            return "technical_failure"
        return "unavailable"

    status = policy_candidate.status
    indicator = policy_candidate.indicator

    if status == "unavailable":
        return "technical_failure"
    if status == "conflict":
        return "semantic_abstention"
    if indicator == "insufficient_evidence":
        return "semantic_abstention"
    if status == "ok" and indicator in {
        "low_concern",
        "moderate_concern",
        "high_concern",
    }:
        return "available"
    return "unavailable"


def _technical_fallback_reason(
    *,
    shadow_enabled: bool,
    shadow_status: Optional[str],
    policy_candidate: Optional[WellbeingPolicyCandidate],
    outcome_kind: WellbeingCandidateOutcomeKind,
    candidate_activated: bool,
    mode: WellbeingAuthorityMode,
) -> Optional[str]:
    if mode == "candidate" and not candidate_activated:
        return "candidate_mode_not_activated"
    if not shadow_enabled:
        return "shadow_disabled"
    if policy_candidate is None:
        if shadow_status == "classifier_unavailable":
            return "classifier_unavailable"
        if shadow_status == "error":
            return "classifier_error"
        return "policy_unavailable"
    if outcome_kind == "technical_failure":
        reasons = set(policy_candidate.reason_codes)
        if "global_unavailable" in reasons or policy_candidate.status == "unavailable":
            if shadow_status == "classifier_unavailable":
                return "classifier_unavailable"
            if shadow_status == "error":
                return "classifier_error"
            return "classifier_unavailable"
        return "policy_unavailable"
    return None


def compare_wellbeing_policies(
    *,
    legacy_indicator: Optional[str],
    policy_candidate: Optional[WellbeingPolicyCandidate],
    mode: Optional[str] = None,
    shadow_enabled: bool = True,
    shadow_status: Optional[str] = None,
    candidate_authority_activated: Optional[bool] = None,
    migration_scope: str = "video",
) -> WellbeingAuthorityComparison:
    """Compare legacy indicator vs policy candidate without mutating either.

    Phase 4C.5: never authorizes candidate into FinalTemporalAssessment.
    ``affects_final_assessment`` is always False.
    """
    resolved_mode: WellbeingAuthorityMode
    if mode is None:
        resolved_mode = resolve_wellbeing_authority_mode()
    else:
        resolved_mode = resolve_wellbeing_authority_mode(override=mode)

    if migration_scope != "video":
        # Control plane is video-first; non-video is unsupported for authority.
        return WellbeingAuthorityComparison(
            mode=resolved_mode,
            migration_scope="video",
            legacy_indicator=legacy_indicator,
            candidate_indicator=None,
            candidate_status=None,
            agreement="candidate_unavailable",
            legacy_available=legacy_indicator is not None,
            candidate_available=False,
            candidate_outcome_kind="unavailable",
            fallback_required=True,
            fallback_reason="unsupported_modality",
            comparison_reason_codes=["unsupported_modality"],
            affects_final_assessment=False,
        )

    activated = (
        resolve_wellbeing_candidate_authority_activated()
        if candidate_authority_activated is None
        else bool(candidate_authority_activated)
    )

    legacy_available = legacy_indicator is not None
    outcome_kind = classify_candidate_outcome(
        policy_candidate,
        shadow_enabled=shadow_enabled,
        shadow_status=shadow_status,
    )

    candidate_indicator: Optional[str] = None
    candidate_status: Optional[str] = None
    candidate_version: Optional[str] = None
    evidence_ids: list[str] = []
    candidate_available = False

    if policy_candidate is not None and shadow_enabled:
        candidate_indicator = policy_candidate.indicator
        candidate_status = policy_candidate.status
        candidate_version = policy_candidate.policy_version
        evidence_ids = list(
            dict.fromkeys(
                list(policy_candidate.supporting_evidence_ids)
                + list(policy_candidate.conflict_evidence_ids),
            ),
        )
        candidate_available = outcome_kind in {
            "available",
            "semantic_abstention",
        }

    reason_codes: list[str] = []
    agreement: WellbeingAuthorityAgreement = "not_compared"
    fallback_required = False
    fallback_reason: Optional[str] = None

    # Candidate mode reserved — never activate authority in Phase 4C.5.
    if resolved_mode == "candidate" and not activated:
        fallback_required = True
        fallback_reason = "candidate_mode_not_activated"
        reason_codes.append("candidate_mode_not_activated")
        agreement = "not_compared"
        return WellbeingAuthorityComparison(
            mode="candidate",
            migration_scope="video",
            legacy_indicator=legacy_indicator,
            candidate_indicator=candidate_indicator,
            candidate_status=candidate_status,
            agreement=agreement,
            legacy_available=legacy_available,
            candidate_available=candidate_available,
            candidate_outcome_kind=outcome_kind,
            fallback_required=fallback_required,
            fallback_reason=fallback_reason,
            comparison_reason_codes=_unique(reason_codes),
            candidate_policy_version=candidate_version,
            candidate_evidence_ids=evidence_ids,
            affects_final_assessment=False,
            note=(
                "Candidate authority mode requested but not activated "
                "(Phase 4C.5 control plane). Legacy remains authoritative."
            ),
        )

    if not shadow_enabled or policy_candidate is None or outcome_kind in {
        "unavailable",
        "technical_failure",
    }:
        agreement = "candidate_unavailable"
        reason_codes.append("candidate_unavailable")
        tech_reason = _technical_fallback_reason(
            shadow_enabled=shadow_enabled,
            shadow_status=shadow_status,
            policy_candidate=policy_candidate,
            outcome_kind=outcome_kind,
            candidate_activated=activated,
            mode=resolved_mode,
        )
        if tech_reason:
            reason_codes.append(tech_reason)
        # Compare/legacy: legacy stays authoritative — no runtime fallback action.
        # Candidate mode (if ever activated): technical failure would require fallback.
        if resolved_mode == "candidate" and activated:
            fallback_required = True
            fallback_reason = tech_reason or "policy_unavailable"
        return WellbeingAuthorityComparison(
            mode=resolved_mode,
            migration_scope="video",
            legacy_indicator=legacy_indicator,
            candidate_indicator=candidate_indicator,
            candidate_status=candidate_status,
            agreement=agreement,
            legacy_available=legacy_available,
            candidate_available=False,
            candidate_outcome_kind=outcome_kind,
            fallback_required=fallback_required,
            fallback_reason=fallback_reason,
            comparison_reason_codes=_unique(reason_codes),
            candidate_policy_version=candidate_version,
            candidate_evidence_ids=evidence_ids,
            affects_final_assessment=False,
        )

    # Candidate present (available or semantic abstention).
    if outcome_kind == "semantic_abstention":
        if candidate_status == "conflict":
            reason_codes.append("candidate_conflict")
        else:
            reason_codes.append("candidate_insufficient")
        # Semantic abstention is NOT automatic technical fallback.
        if resolved_mode == "candidate" and activated:
            fallback_required = False
            fallback_reason = None

    if not legacy_available:
        agreement = "not_compared"
        reason_codes.append("legacy_unavailable")
    else:
        assert legacy_indicator is not None
        assert candidate_indicator is not None
        if legacy_indicator == candidate_indicator:
            agreement = "same"
            reason_codes.append("indicators_agree")
        else:
            agreement = "different"
            code = disagreement_reason_code(legacy_indicator, candidate_indicator)
            if code:
                reason_codes.append(code)
            if legacy_indicator == "insufficient_evidence":
                reason_codes.append("legacy_insufficient_candidate_available")
            if candidate_indicator == "insufficient_evidence":
                reason_codes.append("legacy_available_candidate_insufficient")

    # Phase 4C.5: comparison never authorizes candidate output.
    return WellbeingAuthorityComparison(
        mode=resolved_mode,
        migration_scope="video",
        legacy_indicator=legacy_indicator,
        candidate_indicator=candidate_indicator,
        candidate_status=candidate_status,
        agreement=agreement,
        legacy_available=legacy_available,
        candidate_available=candidate_available,
        candidate_outcome_kind=outcome_kind,
        fallback_required=fallback_required,
        fallback_reason=fallback_reason,
        comparison_reason_codes=_unique(reason_codes),
        candidate_policy_version=candidate_version,
        candidate_evidence_ids=evidence_ids,
        affects_final_assessment=False,
    )


def attach_authority_comparison(
    *,
    policy_candidate: Optional[WellbeingPolicyCandidate],
    legacy_indicator: Optional[str],
    mode: Optional[str] = None,
    shadow_enabled: bool = True,
    shadow_status: Optional[str] = None,
) -> WellbeingAuthorityComparison:
    """Convenience wrapper for shadow attachment (non-mutating of inputs)."""
    return compare_wellbeing_policies(
        legacy_indicator=legacy_indicator,
        policy_candidate=policy_candidate,
        mode=mode,
        shadow_enabled=shadow_enabled,
        shadow_status=shadow_status,
    )


def _unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out
