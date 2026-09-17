"""Phase 4C.7 — guarded candidate authority readiness (DESIGN ONLY).

Defines activation prerequisites, fail-closed guards, FTA mapping design,
observability contracts, and audit records.

This module MUST NOT:
- route FinalTemporalAssessment through the candidate
- call classifiers / OpenRouter / Whisper
- mutate legacy wellbeing gate behavior
- set WELLBEING_CANDIDATE_AUTHORITY_ACTIVATED

Legacy temporal wellbeing remains authoritative. Legacy is retained for
technical fallback and rollback — it is NOT validation ground truth for
the new policy. Disagreement ≠ candidate failure.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from src.config import (
    resolve_wellbeing_authority_mode,
    resolve_wellbeing_candidate_authority_activated,
)
from src.wellbeing.migration import classify_candidate_outcome
from src.wellbeing.policy_candidate import POLICY_VERSION
from src.wellbeing.schemas import (
    WellbeingAuthorityDecision,
    WellbeingAuthorityObservabilityCounters,
    WellbeingAuthorityReadiness,
    WellbeingAuthorityScope,
    WellbeingAuthoritySource,
    WellbeingCandidateOutcomeKind,
    WellbeingIndicatorDisplayMapping,
    WellbeingPolicyCandidate,
    WellbeingPolicyStabilitySnapshot,
)

READINESS_VERSION = "phase4c7-v1"

# Only explicitly approved policy versions may ever authorize.
APPROVED_CANDIDATE_POLICY_VERSIONS: frozenset[str] = frozenset({POLICY_VERSION})

AUTHORITY_ALLOWED_SCOPES: frozenset[str] = frozenset({"video"})

REQUIRED_READINESS_FLAGS: tuple[str, ...] = (
    "candidate_policy_validated",
    "shadow_replay_validated",
    "live_compare_validated",
    "production_like_compare_validated",
    "runtime_acceptable",
    "technical_fallback_validated",
    "rollback_validated",
    "ui_semantics_approved",
    "observability_ready",
)

# Observed controlled-run eligible windows (diagnostic only; not ground truth).
# Earlier controlled / Phase 4B-style run used window 3; Phase 4C.6 used window 2.
# Both yielded isolated local distress → moderate_concern.
OBSERVED_ELIGIBLE_WINDOW_VARIABILITY: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("earlier_controlled_run", (3,)),
    ("phase4c6_live_compare_reasoner_disabled", (2,)),
)

# Current documented readiness snapshot (Phase 4C.7). NOT inferred from env alone.
CURRENT_READINESS_DEFAULTS: Mapping[str, bool] = {
    "candidate_policy_validated": True,  # Phase 4C.3
    "shadow_replay_validated": True,  # Phase 4C.4
    "live_compare_validated": True,  # Phase 4C.6 plumbing
    "production_like_compare_validated": False,  # reasoner was DISABLED in 4C.6
    "runtime_acceptable": False,  # ~133s shadow; pending explicit decision
    "technical_fallback_validated": True,  # unit/sim contract in 4C.5/4C.6
    "rollback_validated": True,  # config-only rollback exercised in 4C.6
    "ui_semantics_approved": False,
    "observability_ready": False,
}


def evaluate_wellbeing_authority_readiness(
    *,
    overrides: Optional[Mapping[str, bool]] = None,
    readiness_version: str = READINESS_VERSION,
) -> WellbeingAuthorityReadiness:
    """Build an explicit multi-flag readiness record (fail-closed).

    Unknown / malformed overrides for required flags are treated as False.
    """
    flags: dict[str, bool] = {k: bool(CURRENT_READINESS_DEFAULTS[k]) for k in REQUIRED_READINESS_FLAGS}
    if overrides:
        for key, value in overrides.items():
            if key not in REQUIRED_READINESS_FLAGS:
                # Unknown keys are ignored for flag computation but block activation.
                continue
            if not isinstance(value, bool):
                flags[key] = False
            else:
                flags[key] = value

    blocking: list[str] = []
    if readiness_version != READINESS_VERSION:
        blocking.append("unknown_or_unapproved_readiness_version")

    if overrides:
        for key, value in overrides.items():
            if key not in REQUIRED_READINESS_FLAGS:
                blocking.append(f"malformed_readiness_unknown_flag:{key}")
            elif not isinstance(value, bool):
                blocking.append(f"malformed_readiness_non_bool:{key}")

    for key in REQUIRED_READINESS_FLAGS:
        if not flags[key]:
            blocking.append(f"missing_or_false:{key}")

    # Hard blockers called out by name for operators.
    if not flags["production_like_compare_validated"]:
        if "production_like_compare_required" not in blocking:
            blocking.append("production_like_compare_required")
    if not flags["runtime_acceptable"]:
        if "runtime_pending_or_unacceptable" not in blocking:
            blocking.append("runtime_pending_or_unacceptable")

    ready = (
        readiness_version == READINESS_VERSION
        and all(flags[k] for k in REQUIRED_READINESS_FLAGS)
        and not any(r.startswith("malformed_") for r in blocking)
        and "unknown_or_unapproved_readiness_version" not in blocking
    )

    return WellbeingAuthorityReadiness(
        readiness_version=readiness_version,
        candidate_policy_validated=flags["candidate_policy_validated"],
        shadow_replay_validated=flags["shadow_replay_validated"],
        live_compare_validated=flags["live_compare_validated"],
        production_like_compare_validated=flags["production_like_compare_validated"],
        runtime_acceptable=flags["runtime_acceptable"],
        technical_fallback_validated=flags["technical_fallback_validated"],
        rollback_validated=flags["rollback_validated"],
        ui_semantics_approved=flags["ui_semantics_approved"],
        observability_ready=flags["observability_ready"],
        ready_for_activation=ready,
        blocking_reasons=_unique(blocking),
    )


def current_authority_readiness() -> WellbeingAuthorityReadiness:
    """Return the documented current readiness state (NOT READY)."""
    return evaluate_wellbeing_authority_readiness()


def phase4c6_counts_as_production_like_compare() -> bool:
    """Phase 4C.6 used TemporalReasonerConfig(enabled=False) — does not qualify."""
    return False


def is_approved_policy_version(policy_version: Optional[str]) -> bool:
    if policy_version is None or not str(policy_version).strip():
        return False
    return str(policy_version).strip() in APPROVED_CANDIDATE_POLICY_VERSIONS


def is_authority_scope_allowed(scope: str) -> bool:
    return str(scope).strip().lower() in AUTHORITY_ALLOWED_SCOPES


def candidate_success_eligible(
    policy_candidate: Optional[WellbeingPolicyCandidate],
) -> bool:
    """Future success case: ok + concern indicator + grounded evidence + approved version."""
    if policy_candidate is None:
        return False
    if policy_candidate.status != "ok":
        return False
    if policy_candidate.indicator not in {
        "low_concern",
        "moderate_concern",
        "high_concern",
    }:
        return False
    if not policy_candidate.supporting_evidence_ids:
        return False
    if not is_approved_policy_version(policy_candidate.policy_version):
        return False
    return True


def design_fta_indicator_from_candidate(
    policy_candidate: WellbeingPolicyCandidate,
) -> Optional[str]:
    """Design-only mapping: candidate indicator → FTA indicator enum.

    Internal enums already align (low/moderate/high/insufficient).
    Does NOT mutate FinalTemporalAssessment. Display labels are separate.
    """
    return policy_candidate.indicator


def recommended_display_mapping() -> WellbeingIndicatorDisplayMapping:
    """Future UI terminology (not wired; not approved)."""
    return WellbeingIndicatorDisplayMapping(approved=False)


def evaluate_candidate_authority_guard(
    *,
    requested_mode: Optional[str] = None,
    candidate_authority_activated: Optional[bool] = None,
    readiness: Optional[WellbeingAuthorityReadiness] = None,
    policy_candidate: Optional[WellbeingPolicyCandidate] = None,
    legacy_indicator: Optional[str] = None,
    migration_scope: str = "video",
    shadow_enabled: bool = True,
    shadow_status: Optional[str] = None,
) -> WellbeingAuthorityDecision:
    """Fail-closed activation guard (DESIGN). Never authorizes FTA in 4C.7.

    Even when ``WELLBEING_AUTHORITY_MODE=candidate``, authority is blocked
    unless activation flag AND readiness are both satisfied. Phase 4C.7
    always returns ``affects_final_assessment=False`` and never selects
    candidate as ``authority_source`` for client output.
    """
    if requested_mode is None:
        resolved_requested = resolve_wellbeing_authority_mode()
    else:
        resolved_requested = resolve_wellbeing_authority_mode(override=requested_mode)

    activated = (
        resolve_wellbeing_candidate_authority_activated()
        if candidate_authority_activated is None
        else bool(candidate_authority_activated)
    )

    readiness_obj = readiness if readiness is not None else current_authority_readiness()
    blocking: list[str] = list(readiness_obj.blocking_reasons)

    scope = str(migration_scope).strip().lower()
    outcome = classify_candidate_outcome(
        policy_candidate,
        shadow_enabled=shadow_enabled,
        shadow_status=shadow_status,
    )

    evidence_ids: list[str] = []
    cand_status: Optional[str] = None
    cand_indicator: Optional[str] = None
    cand_version: Optional[str] = None
    if policy_candidate is not None:
        cand_status = policy_candidate.status
        cand_indicator = policy_candidate.indicator
        cand_version = policy_candidate.policy_version
        evidence_ids = list(
            dict.fromkeys(
                list(policy_candidate.supporting_evidence_ids)
                + list(policy_candidate.conflict_evidence_ids),
            ),
        )

    # --- Fail closed checks (order matters for clear reasons) ---
    if resolved_requested != "candidate":
        # Design path: non-candidate modes never authorize candidate.
        return WellbeingAuthorityDecision(
            requested_mode=resolved_requested,
            resolved_mode=resolved_requested,
            authority_source="legacy",
            migration_scope="video",
            candidate_policy_version=cand_version,
            candidate_status=cand_status,
            candidate_indicator=cand_indicator,
            legacy_indicator_if_available=legacy_indicator,
            fallback_used=False,
            fallback_reason=None,
            candidate_failure_kind=outcome if outcome == "technical_failure" else None,
            legacy_used_as_technical_fallback=False,
            readiness_version=readiness_obj.readiness_version,
            evidence_ids=evidence_ids,
            blocking_reasons=[],
            affects_final_assessment=False,
            note=(
                "Non-candidate mode: legacy remains authoritative. "
                "Phase 4C.7 does not route FinalTemporalAssessment."
            ),
        )

    # requested_mode == candidate — VIDEO-only authority scope
    if scope not in AUTHORITY_ALLOWED_SCOPES:
        blocking.append("unsupported_modality_scope")
        return _blocked_decision(
            requested_mode="candidate",
            resolved_mode="legacy",
            authority_source="blocked",
            scope=scope if scope in {"video", "text", "image", "audio"} else "video",
            policy_candidate=policy_candidate,
            legacy_indicator=legacy_indicator,
            evidence_ids=evidence_ids,
            blocking=blocking,
            outcome=outcome,
            extra_note="Candidate authority is VIDEO-only.",
        )

    if not activated:
        blocking.append("candidate_authority_activation_flag_false")
        return _blocked_decision(
            requested_mode="candidate",
            resolved_mode="legacy",
            authority_source="blocked",
            scope="video",
            policy_candidate=policy_candidate,
            legacy_indicator=legacy_indicator,
            evidence_ids=evidence_ids,
            blocking=blocking,
            outcome=outcome,
            extra_note=(
                "WELLBEING_AUTHORITY_MODE=candidate refused: "
                "WELLBEING_CANDIDATE_AUTHORITY_ACTIVATED is False."
            ),
        )

    if not readiness_obj.ready_for_activation:
        blocking.append("readiness_not_satisfied")
        return _blocked_decision(
            requested_mode="candidate",
            resolved_mode="legacy",
            authority_source="blocked",
            scope="video",
            policy_candidate=policy_candidate,
            legacy_indicator=legacy_indicator,
            evidence_ids=evidence_ids,
            blocking=blocking,
            outcome=outcome,
            extra_note="Candidate mode + activation true but readiness false → blocked.",
        )

    if not is_approved_policy_version(cand_version):
        if cand_version is None or not str(cand_version).strip():
            blocking.append("missing_policy_version")
        else:
            blocking.append("unapproved_policy_version")
        return _blocked_decision(
            requested_mode="candidate",
            resolved_mode="legacy",
            authority_source="blocked",
            scope="video",
            policy_candidate=policy_candidate,
            legacy_indicator=legacy_indicator,
            evidence_ids=evidence_ids,
            blocking=blocking,
            outcome=outcome,
            extra_note="Missing or unapproved policy version → fail closed.",
        )

    if policy_candidate is None or not evidence_ids:
        blocking.append("missing_evidence_context")
        return _blocked_decision(
            requested_mode="candidate",
            resolved_mode="legacy",
            authority_source="blocked",
            scope="video",
            policy_candidate=policy_candidate,
            legacy_indicator=legacy_indicator,
            evidence_ids=evidence_ids,
            blocking=blocking,
            outcome=outcome,
            extra_note="Missing evidence context → fail closed.",
        )

    # Phase 4C.7 HARD STOP: even if all gates pass in a hypothetical override,
    # this design phase never authorizes candidate into FTA.
    blocking.append("phase4c7_design_only_activation_not_wired")
    return _blocked_decision(
        requested_mode="candidate",
        resolved_mode="legacy",
        authority_source="blocked",
        scope="video",
        policy_candidate=policy_candidate,
        legacy_indicator=legacy_indicator,
        evidence_ids=evidence_ids,
        blocking=blocking,
        outcome=outcome,
        extra_note=(
            "Phase 4C.7 design-only: activation path is defined but not wired. "
            "FinalTemporalAssessment remains legacy-controlled."
        ),
    )


def future_technical_failure_fallback_decision(
    *,
    policy_candidate: Optional[WellbeingPolicyCandidate],
    legacy_indicator: Optional[str],
    shadow_status: Optional[str] = None,
) -> WellbeingAuthorityDecision:
    """Document FUTURE candidate-authority technical fallback (not activated).

    Semantic abstention must NOT use this path. Technical failure MAY fall
    back to legacy with explicit observability fields.
    """
    outcome = classify_candidate_outcome(
        policy_candidate,
        shadow_enabled=True,
        shadow_status=shadow_status,
    )
    if outcome != "technical_failure":
        return WellbeingAuthorityDecision(
            requested_mode="candidate",
            resolved_mode="candidate",
            authority_source="blocked",
            candidate_status=policy_candidate.status if policy_candidate else None,
            candidate_indicator=(
                policy_candidate.indicator if policy_candidate else None
            ),
            legacy_indicator_if_available=legacy_indicator,
            fallback_used=False,
            candidate_failure_kind=outcome,
            legacy_used_as_technical_fallback=False,
            readiness_version=READINESS_VERSION,
            affects_final_assessment=False,
            note=(
                "Not a technical failure — no technical fallback. "
                "Semantic abstention preserves Insufficient Evidence."
            ),
        )

    return WellbeingAuthorityDecision(
        requested_mode="candidate",
        resolved_mode="legacy",
        authority_source="legacy_technical_fallback",
        candidate_policy_version=(
            policy_candidate.policy_version if policy_candidate else None
        ),
        candidate_status=policy_candidate.status if policy_candidate else None,
        candidate_indicator=(
            policy_candidate.indicator if policy_candidate else None
        ),
        legacy_indicator_if_available=legacy_indicator,
        fallback_used=True,
        fallback_reason=_technical_fallback_reason(shadow_status, policy_candidate),
        candidate_failure_kind="technical_failure",
        legacy_used_as_technical_fallback=True,
        readiness_version=READINESS_VERSION,
        affects_final_assessment=False,
        note=(
            "FUTURE candidate-authority contract only: technical failure may "
            "fall back to legacy with explicit flags. Not wired in Phase 4C.7."
        ),
    )


def future_semantic_abstention_preserves_insufficient(
    policy_candidate: WellbeingPolicyCandidate,
) -> bool:
    """Semantic insufficient/conflict → keep Insufficient Evidence (not failure)."""
    outcome = classify_candidate_outcome(policy_candidate)
    return outcome == "semantic_abstention" and (
        policy_candidate.indicator == "insufficient_evidence"
        or policy_candidate.status in {"insufficient_evidence", "conflict"}
    )


def build_policy_stability_snapshot(
    *,
    run_id: str,
    policy_candidate: Optional[WellbeingPolicyCandidate],
    eligible_window_indices: Sequence[int] = (),
) -> WellbeingPolicyStabilitySnapshot:
    """Record policy-level fields; window indices are diagnostic only."""
    if policy_candidate is None:
        return WellbeingPolicyStabilitySnapshot(
            run_id=run_id,
            eligible_window_indices=list(eligible_window_indices),
        )
    return WellbeingPolicyStabilitySnapshot(
        run_id=run_id,
        candidate_indicator=policy_candidate.indicator,
        candidate_status=policy_candidate.status,
        local_support_level=policy_candidate.local_support_level,
        distress_pattern=policy_candidate.distress_pattern,
        recovery_pattern=policy_candidate.recovery_pattern,
        eligible_window_indices=list(eligible_window_indices),
        policy_version=policy_candidate.policy_version,
    )


def policy_level_semantics_stable(
    a: WellbeingPolicyStabilitySnapshot,
    b: WellbeingPolicyStabilitySnapshot,
) -> bool:
    """True when policy semantics match; eligible window indices may differ."""
    return (
        a.candidate_indicator == b.candidate_indicator
        and a.candidate_status == b.candidate_status
        and a.local_support_level == b.local_support_level
        and a.distress_pattern == b.distress_pattern
        and a.recovery_pattern == b.recovery_pattern
        and a.policy_version == b.policy_version
    )


def empty_observability_counters() -> WellbeingAuthorityObservabilityCounters:
    return WellbeingAuthorityObservabilityCounters()


def rollback_authority_mode() -> str:
    """Operational rollback: set WELLBEING_AUTHORITY_MODE=legacy (config only)."""
    return "legacy"


def _technical_fallback_reason(
    shadow_status: Optional[str],
    policy_candidate: Optional[WellbeingPolicyCandidate],
) -> str:
    if shadow_status == "classifier_unavailable":
        return "classifier_unavailable"
    if shadow_status == "error":
        return "classifier_error"
    if policy_candidate is None:
        return "policy_unavailable"
    if policy_candidate.status == "unavailable":
        return "policy_unavailable"
    return "policy_unavailable"


def _blocked_decision(
    *,
    requested_mode: str,
    resolved_mode: str,
    authority_source: WellbeingAuthoritySource,
    scope: str,
    policy_candidate: Optional[WellbeingPolicyCandidate],
    legacy_indicator: Optional[str],
    evidence_ids: list[str],
    blocking: Sequence[str],
    outcome: WellbeingCandidateOutcomeKind,
    extra_note: str,
) -> WellbeingAuthorityDecision:
    scope_lit: WellbeingAuthorityScope = (
        scope if scope in {"video", "text", "image", "audio"} else "video"  # type: ignore[assignment]
    )
    return WellbeingAuthorityDecision(
        requested_mode=requested_mode,  # type: ignore[arg-type]
        resolved_mode=resolved_mode,  # type: ignore[arg-type]
        authority_source=authority_source,
        migration_scope=scope_lit,
        candidate_policy_version=(
            policy_candidate.policy_version if policy_candidate else None
        ),
        candidate_status=policy_candidate.status if policy_candidate else None,
        candidate_indicator=(
            policy_candidate.indicator if policy_candidate else None
        ),
        legacy_indicator_if_available=legacy_indicator,
        fallback_used=False,
        fallback_reason=None,
        candidate_failure_kind=outcome if outcome == "technical_failure" else None,
        legacy_used_as_technical_fallback=False,
        readiness_version=READINESS_VERSION,
        evidence_ids=list(evidence_ids),
        blocking_reasons=_unique(blocking),
        affects_final_assessment=False,
        note=extra_note,
    )


def _unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


# Runtime documentation (no implementation).
RUNTIME_NOTES = {
    "phase4c6_whole_pipeline_seconds": 170.7,
    "phase4c6_shadow_seconds": 133.0,
    "interactive_latency_suitable": False,
    "eod_batch_throughput_suitable": "pending_volume_analysis",
    "possible_later_paths": (
        "dedicated_inference_service",
        "gpu_execution",
        "caching",
        "asynchronous_eod_batch",
        "batching_across_posts_videos",
        "model_or_runtime_optimization",
    ),
}

CONFIG_MATRIX_NOTES = """
shadow=false + legacy → legacy only
shadow=true + legacy → legacy + shadow diagnostics
shadow=true + compare → legacy authoritative + candidate comparison
shadow=true + candidate + activation=false → candidate authority BLOCKED
shadow=true + candidate + activation=true + readiness=false → BLOCKED
shadow=true + candidate + activation=true + readiness=true → FUTURE authoritative path (not wired in 4C.7)
"""
