"""Phase 4C.7 authority activation readiness design tests (no ML / no video)."""

from __future__ import annotations

import inspect

import pytest

from src.config import (
    WELLBEING_AUTHORITY_MODE,
    WELLBEING_CANDIDATE_AUTHORITY_ACTIVATED,
    resolve_wellbeing_authority_mode,
)
from src.schemas import FinalTemporalAssessment
from src.wellbeing.authority_readiness import (
    APPROVED_CANDIDATE_POLICY_VERSIONS,
    READINESS_VERSION,
    REQUIRED_READINESS_FLAGS,
    build_policy_stability_snapshot,
    candidate_success_eligible,
    current_authority_readiness,
    design_fta_indicator_from_candidate,
    evaluate_candidate_authority_guard,
    evaluate_wellbeing_authority_readiness,
    future_semantic_abstention_preserves_insufficient,
    future_technical_failure_fallback_decision,
    is_approved_policy_version,
    is_authority_scope_allowed,
    phase4c6_counts_as_production_like_compare,
    policy_level_semantics_stable,
    recommended_display_mapping,
    rollback_authority_mode,
)
from src.wellbeing.policy_candidate import POLICY_VERSION
from src.wellbeing.schemas import WellbeingPolicyCandidate


def _all_true() -> dict[str, bool]:
    return {k: True for k in REQUIRED_READINESS_FLAGS}


def _candidate(
    *,
    indicator: str = "moderate_concern",
    status: str = "ok",
    version: str = POLICY_VERSION,
    evidence: list[str] | None = None,
) -> WellbeingPolicyCandidate:
    return WellbeingPolicyCandidate(
        status=status,  # type: ignore[arg-type]
        indicator=indicator,  # type: ignore[arg-type]
        reason_codes=["global_personal_distress"],
        supporting_evidence_ids=list(
            evidence if evidence is not None else ["wellbeing-global-transcript"]
        ),
        conflict_evidence_ids=[],
        global_eligible=True,
        local_support_level="isolated",
        distress_pattern="isolated",
        recovery_pattern="none",
        policy_version=version,  # type: ignore[arg-type]
        affects_final_assessment=False,
    )


def test_01_current_state_not_ready() -> None:
    r = current_authority_readiness()
    assert r.ready_for_activation is False
    assert r.production_like_compare_validated is False
    assert r.runtime_acceptable is False
    assert r.ui_semantics_approved is False
    assert r.observability_ready is False
    assert WELLBEING_CANDIDATE_AUTHORITY_ACTIVATED is False
    assert WELLBEING_AUTHORITY_MODE == "legacy"


def test_02_missing_production_like_compare_blocks() -> None:
    overrides = _all_true()
    overrides["production_like_compare_validated"] = False
    r = evaluate_wellbeing_authority_readiness(overrides=overrides)
    assert r.ready_for_activation is False
    assert "missing_or_false:production_like_compare_validated" in r.blocking_reasons
    assert "production_like_compare_required" in r.blocking_reasons
    assert phase4c6_counts_as_production_like_compare() is False


def test_03_runtime_pending_blocks() -> None:
    overrides = _all_true()
    overrides["runtime_acceptable"] = False
    r = evaluate_wellbeing_authority_readiness(overrides=overrides)
    assert r.ready_for_activation is False
    assert "runtime_pending_or_unacceptable" in r.blocking_reasons


def test_04_missing_rollback_blocks() -> None:
    overrides = _all_true()
    overrides["rollback_validated"] = False
    r = evaluate_wellbeing_authority_readiness(overrides=overrides)
    assert r.ready_for_activation is False
    assert "missing_or_false:rollback_validated" in r.blocking_reasons


def test_05_missing_observability_blocks() -> None:
    overrides = _all_true()
    overrides["observability_ready"] = False
    r = evaluate_wellbeing_authority_readiness(overrides=overrides)
    assert r.ready_for_activation is False
    assert "missing_or_false:observability_ready" in r.blocking_reasons


def test_06_all_prerequisites_true_readiness_true() -> None:
    r = evaluate_wellbeing_authority_readiness(overrides=_all_true())
    assert r.ready_for_activation is True
    assert r.blocking_reasons == []
    assert r.readiness_version == READINESS_VERSION


def test_07_candidate_mode_activation_false_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WELLBEING_AUTHORITY_MODE", "candidate")
    monkeypatch.setenv("WELLBEING_CANDIDATE_AUTHORITY_ACTIVATED", "0")
    decision = evaluate_candidate_authority_guard(
        requested_mode="candidate",
        candidate_authority_activated=False,
        readiness=evaluate_wellbeing_authority_readiness(overrides=_all_true()),
        policy_candidate=_candidate(),
        legacy_indicator="insufficient_evidence",
        migration_scope="video",
    )
    assert decision.authority_source == "blocked"
    assert decision.affects_final_assessment is False
    assert "candidate_authority_activation_flag_false" in decision.blocking_reasons


def test_08_candidate_mode_readiness_false_blocked() -> None:
    decision = evaluate_candidate_authority_guard(
        requested_mode="candidate",
        candidate_authority_activated=True,
        readiness=current_authority_readiness(),
        policy_candidate=_candidate(),
        legacy_indicator="moderate_concern",
        migration_scope="video",
    )
    assert decision.authority_source == "blocked"
    assert decision.resolved_mode == "legacy"
    assert "readiness_not_satisfied" in decision.blocking_reasons
    assert decision.affects_final_assessment is False


def test_09_malformed_readiness_blocked() -> None:
    r = evaluate_wellbeing_authority_readiness(
        overrides={**_all_true(), "not_a_real_flag": True},  # type: ignore[dict-item]
    )
    assert r.ready_for_activation is False
    assert any(x.startswith("malformed_readiness_unknown_flag") for x in r.blocking_reasons)


def test_10_unknown_policy_version_blocked() -> None:
    # Bypass Literal validation to simulate an unapproved runtime version.
    bad = WellbeingPolicyCandidate.model_construct(
        status="ok",
        indicator="moderate_concern",
        reason_codes=["global_personal_distress"],
        supporting_evidence_ids=["wellbeing-global-transcript"],
        conflict_evidence_ids=[],
        global_eligible=True,
        local_support_level="isolated",
        distress_pattern="isolated",
        recovery_pattern="none",
        policy_version="phase4c2-v999",
        affects_final_assessment=False,
    )
    decision = evaluate_candidate_authority_guard(
        requested_mode="candidate",
        candidate_authority_activated=True,
        readiness=evaluate_wellbeing_authority_readiness(overrides=_all_true()),
        policy_candidate=bad,
        migration_scope="video",
    )
    assert decision.authority_source == "blocked"
    assert "unapproved_policy_version" in decision.blocking_reasons
    assert is_approved_policy_version("phase4c2-v999") is False


def test_11_approved_policy_version_accepted() -> None:
    assert POLICY_VERSION in APPROVED_CANDIDATE_POLICY_VERSIONS
    assert is_approved_policy_version(POLICY_VERSION) is True
    assert candidate_success_eligible(_candidate()) is True


def test_12_semantic_abstention_not_technical_failure() -> None:
    cand = _candidate(indicator="insufficient_evidence", status="insufficient_evidence")
    assert future_semantic_abstention_preserves_insufficient(cand) is True
    fb = future_technical_failure_fallback_decision(
        policy_candidate=cand,
        legacy_indicator="low_concern",
    )
    assert fb.fallback_used is False
    assert fb.legacy_used_as_technical_fallback is False


def test_13_technical_failure_eligible_for_fallback() -> None:
    cand = _candidate(status="unavailable", indicator="insufficient_evidence")
    fb = future_technical_failure_fallback_decision(
        policy_candidate=cand,
        legacy_indicator="moderate_concern",
        shadow_status="classifier_unavailable",
    )
    assert fb.fallback_used is True
    assert fb.legacy_used_as_technical_fallback is True
    assert fb.authority_source == "legacy_technical_fallback"
    assert fb.fallback_reason == "classifier_unavailable"
    assert fb.affects_final_assessment is False


@pytest.mark.parametrize(
    "indicator",
    ["low_concern", "moderate_concern", "high_concern", "insufficient_evidence"],
)
def test_14_17_candidate_mapping_design(indicator: str) -> None:
    status = "ok" if indicator != "insufficient_evidence" else "insufficient_evidence"
    cand = _candidate(indicator=indicator, status=status)
    assert design_fta_indicator_from_candidate(cand) == indicator
    display = recommended_display_mapping()
    assert display.product_label == "Wellbeing Indicator"
    assert display.approved is False
    assert "Stress" not in display.low_concern
    assert display.low_concern == "Low Concern"
    assert display.moderate_concern == "Moderate Concern"
    assert display.high_concern == "High Concern"
    assert display.insufficient_evidence == "Insufficient Evidence"


def test_18_video_scope_allowed() -> None:
    assert is_authority_scope_allowed("video") is True


@pytest.mark.parametrize("scope", ["text", "image", "audio"])
def test_19_21_non_video_scope_blocked(scope: str) -> None:
    assert is_authority_scope_allowed(scope) is False
    decision = evaluate_candidate_authority_guard(
        requested_mode="candidate",
        candidate_authority_activated=True,
        readiness=evaluate_wellbeing_authority_readiness(overrides=_all_true()),
        policy_candidate=_candidate(),
        migration_scope=scope,
    )
    assert decision.authority_source == "blocked"
    assert "unsupported_modality_scope" in decision.blocking_reasons


def test_22_rollback_resolves_to_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WELLBEING_AUTHORITY_MODE", "compare")
    assert resolve_wellbeing_authority_mode() == "compare"
    monkeypatch.setenv("WELLBEING_AUTHORITY_MODE", rollback_authority_mode())
    assert resolve_wellbeing_authority_mode() == "legacy"
    # Old gate remains importable — no code revert.
    from src.temporal import wellbeing as legacy_gate

    assert hasattr(legacy_gate, "compute_wellbeing_indicator")


def test_23_24_25_26_no_forbidden_dependencies() -> None:
    import src.wellbeing.authority_readiness as mod

    src = inspect.getsource(mod)
    assert "get_wellbeing_classifier" not in src
    assert "classify_many" not in src
    assert "from src.temporal.providers" not in src
    assert "import src.openrouter" not in src
    assert "openrouter_api" not in src.lower()
    assert "siglip" not in src.lower()
    assert "tesseract" not in src.lower()
    assert "faster_whisper" not in src.lower()
    assert "build_wellbeing_shadow" not in src
    # No visual/OCR feature inputs in signatures or logic.
    sig = inspect.signature(evaluate_candidate_authority_guard)
    for banned in ("transcript", "ocr", "visual", "faces", "sentiment"):
        assert banned not in sig.parameters


def test_27_no_numerical_score() -> None:
    r = current_authority_readiness()
    blob = str(r.model_dump()).lower()
    for bad in ("wellbeing_score", "mental_health_score", "0-100", "severity_score"):
        assert bad not in blob
    assert "score" not in WellbeingPolicyCandidate.model_fields


def test_28_no_authority_change_to_fta() -> None:
    fta = FinalTemporalAssessment(
        overall_wellbeing_indicator="insufficient_evidence",
        status="disabled",
    )
    before = fta.model_dump()
    decision = evaluate_candidate_authority_guard(
        requested_mode="candidate",
        candidate_authority_activated=True,
        readiness=evaluate_wellbeing_authority_readiness(overrides=_all_true()),
        policy_candidate=_candidate(indicator="high_concern"),
        legacy_indicator=fta.overall_wellbeing_indicator,
        migration_scope="video",
    )
    assert fta.model_dump() == before
    assert decision.affects_final_assessment is False
    assert "authority_decision" not in FinalTemporalAssessment.model_fields
    assert "policy_candidate" not in FinalTemporalAssessment.model_fields


def test_29_authority_decision_no_raw_text() -> None:
    decision = evaluate_candidate_authority_guard(
        requested_mode="candidate",
        candidate_authority_activated=False,
        policy_candidate=_candidate(),
    )
    keys: set[str] = set()

    def walk(obj: object) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                keys.add(str(k))
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(decision.model_dump())
    for bad in ("text", "transcript", "raw_text", "speech_segments", "ocr_text", "caption"):
        assert bad not in keys


def test_30_activation_fails_closed() -> None:
    # Even with activation + full readiness + approved candidate, 4C.7 stays blocked.
    decision = evaluate_candidate_authority_guard(
        requested_mode="candidate",
        candidate_authority_activated=True,
        readiness=evaluate_wellbeing_authority_readiness(overrides=_all_true()),
        policy_candidate=_candidate(),
        migration_scope="video",
    )
    assert decision.authority_source == "blocked"
    assert decision.affects_final_assessment is False
    assert "phase4c7_design_only_activation_not_wired" in decision.blocking_reasons

    # Missing policy version
    empty = _candidate(evidence=[])
    decision2 = evaluate_candidate_authority_guard(
        requested_mode="candidate",
        candidate_authority_activated=True,
        readiness=evaluate_wellbeing_authority_readiness(overrides=_all_true()),
        policy_candidate=empty,
        migration_scope="video",
    )
    assert decision2.authority_source == "blocked"
    assert "missing_evidence_context" in decision2.blocking_reasons


def test_policy_stability_window_index_may_differ() -> None:
    cand = _candidate()
    a = build_policy_stability_snapshot(
        run_id="earlier",
        policy_candidate=cand,
        eligible_window_indices=[3],
    )
    b = build_policy_stability_snapshot(
        run_id="phase4c6",
        policy_candidate=cand,
        eligible_window_indices=[2],
    )
    assert a.eligible_window_indices != b.eligible_window_indices
    assert policy_level_semantics_stable(a, b) is True


def test_missing_policy_version_blocks() -> None:
    assert is_approved_policy_version(None) is False
    assert is_approved_policy_version("") is False
