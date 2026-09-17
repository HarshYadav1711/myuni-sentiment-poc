"""Phase 4C.5 wellbeing authority migration control-plane tests (no ML)."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

from src.config import (
    WELLBEING_AUTHORITY_MODE,
    WELLBEING_CANDIDATE_AUTHORITY_ACTIVATED,
    resolve_wellbeing_authority_mode,
    resolve_wellbeing_candidate_authority_activated,
    resolve_wellbeing_shadow_enabled,
)
from src.pipeline import MyUniSentimentPipeline
from src.schemas import AnalysisBlock, FinalTemporalAssessment, ModalityBundle, SentimentEvidence
from src.wellbeing.migration import (
    classify_candidate_outcome,
    compare_wellbeing_policies,
    disagreement_reason_code,
)
from src.wellbeing.schemas import WellbeingPolicyCandidate, WellbeingShadowAnalysis


def _ev() -> SentimentEvidence:
    return SentimentEvidence(
        label="neutral",
        score=0.0,
        confidence=0.5,
        probabilities={"negative": 0.2, "neutral": 0.6, "positive": 0.2},
        model="stub",
    )


def _candidate(
    *,
    indicator: str = "moderate_concern",
    status: str = "ok",
    reasons: list[str] | None = None,
    evidence_ids: list[str] | None = None,
) -> WellbeingPolicyCandidate:
    return WellbeingPolicyCandidate(
        status=status,  # type: ignore[arg-type]
        indicator=indicator,  # type: ignore[arg-type]
        reason_codes=list(reasons or []),
        supporting_evidence_ids=list(evidence_ids or ["wellbeing-global-transcript"]),
        policy_version="phase4c2-v1",
        affects_final_assessment=False,
    )


def _fta(indicator: str = "moderate_concern") -> FinalTemporalAssessment:
    return FinalTemporalAssessment(overall_wellbeing_indicator=indicator)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_01_default_mode_legacy() -> None:
    assert WELLBEING_AUTHORITY_MODE == "legacy"
    assert resolve_wellbeing_authority_mode() == "legacy"
    assert WELLBEING_CANDIDATE_AUTHORITY_ACTIVATED is False
    assert resolve_wellbeing_candidate_authority_activated() is False


def test_02_valid_mode_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    for mode in ("legacy", "compare", "candidate"):
        monkeypatch.setenv("WELLBEING_AUTHORITY_MODE", mode)
        assert resolve_wellbeing_authority_mode() == mode
    assert resolve_wellbeing_authority_mode(override="compare") == "compare"


def test_03_invalid_mode_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WELLBEING_AUTHORITY_MODE", "canddate")
    with pytest.raises(ValueError, match="Invalid WELLBEING_AUTHORITY_MODE"):
        resolve_wellbeing_authority_mode()
    with pytest.raises(ValueError):
        resolve_wellbeing_authority_mode(override="nope")


def test_04_legacy_mode_unchanged_authority() -> None:
    cand = _candidate(indicator="high_concern")
    cmp = compare_wellbeing_policies(
        legacy_indicator="moderate_concern",
        policy_candidate=cand,
        mode="legacy",
    )
    assert cmp.mode == "legacy"
    assert cmp.affects_final_assessment is False
    assert cmp.agreement == "different"
    assert cmp.fallback_required is False


def test_05_compare_mode_unchanged_authority() -> None:
    cand = _candidate(indicator="high_concern")
    cmp = compare_wellbeing_policies(
        legacy_indicator="moderate_concern",
        policy_candidate=cand,
        mode="compare",
    )
    assert cmp.mode == "compare"
    assert cmp.affects_final_assessment is False
    assert cmp.agreement == "different"
    # Compare never requires fallback action — legacy stays authoritative.
    assert cmp.fallback_required is False


@pytest.mark.parametrize(
    ("legacy", "candidate"),
    [
        ("low_concern", "low_concern"),
        ("moderate_concern", "moderate_concern"),
        ("high_concern", "high_concern"),
        ("insufficient_evidence", "insufficient_evidence"),
    ],
)
def test_06_same_indicator_comparison(legacy: str, candidate: str) -> None:
    cmp = compare_wellbeing_policies(
        legacy_indicator=legacy,
        policy_candidate=_candidate(indicator=candidate),
        mode="compare",
    )
    assert cmp.agreement == "same"
    assert "indicators_agree" in cmp.comparison_reason_codes


@pytest.mark.parametrize(
    ("legacy", "candidate", "code"),
    [
        ("low_concern", "moderate_concern", "legacy_low_candidate_moderate"),
        ("moderate_concern", "low_concern", "legacy_moderate_candidate_low"),
        ("high_concern", "moderate_concern", "legacy_high_candidate_moderate"),
        ("moderate_concern", "high_concern", "legacy_moderate_candidate_high"),
        ("insufficient_evidence", "moderate_concern", "legacy_insufficient_candidate_moderate"),
        ("moderate_concern", "insufficient_evidence", "legacy_moderate_candidate_insufficient"),
    ],
)
def test_07_different_indicator_comparison(legacy: str, candidate: str, code: str) -> None:
    status = "ok" if candidate != "insufficient_evidence" else "insufficient_evidence"
    if candidate == "insufficient_evidence":
        cand = _candidate(indicator=candidate, status="insufficient_evidence")
    else:
        cand = _candidate(indicator=candidate, status=status)
    cmp = compare_wellbeing_policies(
        legacy_indicator=legacy,
        policy_candidate=cand,
        mode="compare",
    )
    assert cmp.agreement == "different"
    assert code in cmp.comparison_reason_codes
    assert disagreement_reason_code(legacy, candidate) == code


def test_08_candidate_technical_unavailable() -> None:
    cmp = compare_wellbeing_policies(
        legacy_indicator="moderate_concern",
        policy_candidate=_candidate(status="unavailable", indicator="insufficient_evidence"),
        mode="compare",
        shadow_status="classifier_unavailable",
    )
    assert cmp.agreement == "candidate_unavailable"
    assert cmp.candidate_outcome_kind == "technical_failure"
    assert "candidate_unavailable" in cmp.comparison_reason_codes


def test_09_candidate_conflict() -> None:
    cand = _candidate(
        indicator="insufficient_evidence",
        status="conflict",
        reasons=["mixed_distress_recovery"],
    )
    cmp = compare_wellbeing_policies(
        legacy_indicator="moderate_concern",
        policy_candidate=cand,
        mode="compare",
    )
    assert cmp.candidate_outcome_kind == "semantic_abstention"
    assert "candidate_conflict" in cmp.comparison_reason_codes
    assert cmp.agreement == "different"
    assert cmp.fallback_required is False  # abstention ≠ technical fallback


def test_10_candidate_semantic_insufficient() -> None:
    cand = _candidate(
        indicator="insufficient_evidence",
        status="insufficient_evidence",
        reasons=["global_not_eligible"],
    )
    cmp = compare_wellbeing_policies(
        legacy_indicator="low_concern",
        policy_candidate=cand,
        mode="compare",
    )
    assert cmp.candidate_outcome_kind == "semantic_abstention"
    assert "candidate_insufficient" in cmp.comparison_reason_codes
    assert "legacy_available_candidate_insufficient" in cmp.comparison_reason_codes


def test_11_technical_failure_vs_semantic_abstention() -> None:
    tech = classify_candidate_outcome(
        _candidate(status="unavailable", indicator="insufficient_evidence"),
        shadow_status="error",
    )
    semantic = classify_candidate_outcome(
        _candidate(status="insufficient_evidence", indicator="insufficient_evidence"),
    )
    conflict = classify_candidate_outcome(
        _candidate(status="conflict", indicator="insufficient_evidence"),
    )
    assert tech == "technical_failure"
    assert semantic == "semantic_abstention"
    assert conflict == "semantic_abstention"


def test_12_shadow_disabled_compare() -> None:
    cmp = compare_wellbeing_policies(
        legacy_indicator="moderate_concern",
        policy_candidate=_candidate(),
        mode="compare",
        shadow_enabled=False,
    )
    assert cmp.agreement == "candidate_unavailable"
    assert "shadow_disabled" in cmp.comparison_reason_codes
    assert cmp.candidate_available is False


def test_13_fallback_reason_deterministic() -> None:
    a = compare_wellbeing_policies(
        legacy_indicator="moderate_concern",
        policy_candidate=None,
        mode="candidate",
        candidate_authority_activated=False,
    )
    b = compare_wellbeing_policies(
        legacy_indicator="moderate_concern",
        policy_candidate=None,
        mode="candidate",
        candidate_authority_activated=False,
    )
    assert a.model_dump() == b.model_dump()
    assert a.fallback_reason == "candidate_mode_not_activated"
    assert a.fallback_required is True


def test_14_candidate_policy_version_retained() -> None:
    cmp = compare_wellbeing_policies(
        legacy_indicator="moderate_concern",
        policy_candidate=_candidate(),
        mode="compare",
    )
    assert cmp.candidate_policy_version == "phase4c2-v1"


def test_15_evidence_ids_retained() -> None:
    cmp = compare_wellbeing_policies(
        legacy_indicator="moderate_concern",
        policy_candidate=_candidate(
            evidence_ids=["wellbeing-global-transcript", "wellbeing-window-3"],
        ),
        mode="compare",
    )
    assert "wellbeing-global-transcript" in cmp.candidate_evidence_ids
    assert "wellbeing-window-3" in cmp.candidate_evidence_ids


def test_16_no_raw_text() -> None:
    cmp = compare_wellbeing_policies(
        legacy_indicator="moderate_concern",
        policy_candidate=_candidate(),
        mode="compare",
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

    walk(cmp.model_dump())
    for bad in ("text", "transcript_text", "raw_text", "speech_segments", "utterance"):
        assert bad not in keys


def test_17_comparison_affects_final_assessment_false() -> None:
    cmp = compare_wellbeing_policies(
        legacy_indicator="high_concern",
        policy_candidate=_candidate(indicator="low_concern"),
        mode="compare",
    )
    assert cmp.affects_final_assessment is False


def test_18_rollback_to_legacy_config_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WELLBEING_AUTHORITY_MODE", "compare")
    assert resolve_wellbeing_authority_mode() == "compare"
    monkeypatch.setenv("WELLBEING_AUTHORITY_MODE", "legacy")
    assert resolve_wellbeing_authority_mode() == "legacy"
    # No code path deletion required — temporal wellbeing module still importable.
    from src.temporal import wellbeing as legacy_gate

    assert hasattr(legacy_gate, "compute_wellbeing_indicator")


def test_19_20_21_22_23_no_forbidden_dependencies() -> None:
    import src.wellbeing.migration as mod

    src = inspect.getsource(mod)
    assert "from src.temporal" not in src
    assert "import src.temporal" not in src
    assert "from src.openrouter" not in src
    assert "classify_many" not in src
    assert "get_wellbeing_classifier" not in src
    sig = inspect.signature(compare_wellbeing_policies)
    for banned in ("transcript", "visual", "ocr", "faces", "temporal_features", "sentiment"):
        assert banned not in sig.parameters


def test_24_final_temporal_assessment_legacy_equality() -> None:
    """Attaching comparison must not alter FTA fields."""
    fta = _fta("moderate_concern")
    before = fta.model_dump()
    cmp = compare_wellbeing_policies(
        legacy_indicator=fta.overall_wellbeing_indicator,
        policy_candidate=_candidate(indicator="high_concern"),
        mode="legacy",
    )
    assert fta.model_dump() == before
    assert cmp.affects_final_assessment is False
    assert "policy_candidate" not in FinalTemporalAssessment.model_fields
    assert "authority_comparison" not in FinalTemporalAssessment.model_fields


def test_25_final_temporal_assessment_compare_equality() -> None:
    fta = _fta("low_concern")
    before = fta.model_dump()
    cmp = compare_wellbeing_policies(
        legacy_indicator=fta.overall_wellbeing_indicator,
        policy_candidate=_candidate(indicator="moderate_concern"),
        mode="compare",
    )
    assert fta.model_dump() == before
    assert cmp.mode == "compare"
    assert cmp.agreement == "different"


def test_26_candidate_authority_not_activated() -> None:
    cmp = compare_wellbeing_policies(
        legacy_indicator="moderate_concern",
        policy_candidate=_candidate(indicator="high_concern"),
        mode="candidate",
        candidate_authority_activated=False,
    )
    assert cmp.fallback_required is True
    assert cmp.fallback_reason == "candidate_mode_not_activated"
    assert cmp.agreement == "not_compared"
    assert cmp.affects_final_assessment is False


def test_27_no_numerical_score() -> None:
    cmp = compare_wellbeing_policies(
        legacy_indicator="moderate_concern",
        policy_candidate=_candidate(),
        mode="compare",
    )
    blob = str(cmp.model_dump()).lower()
    for forbidden in ("wellbeing_score", "mental_health_score", "0-100", "severity"):
        assert forbidden not in blob
    assert "score" not in cmp.model_fields


def test_28_terminology_docs_avoid_clinical_claims() -> None:
    from pathlib import Path

    docs = Path("docs/wellbeing_classifier.md").read_text(encoding="utf-8")
    assert "## Phase 4C.5 — authority migration design" in docs
    section = docs.split("## Phase 4C.5 — authority migration design")[1]
    section = section.split("## Why OpenRouter")[0]
    lower = section.lower()
    assert "wellbeing indicator" in lower
    assert "high concern" in lower
    assert "suicide risk" in lower
    assert "does **not** mean" in section or "does not mean" in lower
    assert "not activated" in lower
    assert "clinical emergency" in lower
    assert "wellbeing_authority_mode" in lower


def test_pipeline_shadow_disabled_keeps_fta(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WELLBEING_SHADOW_ENABLED", "0")
    monkeypatch.setenv("WELLBEING_AUTHORITY_MODE", "compare")
    pipe = MyUniSentimentPipeline()
    text_mock = MagicMock()
    text_mock.validate_text.side_effect = lambda t: str(t).strip()
    text_mock.analyze.return_value = _ev()
    pipe._text_analyzer = text_mock
    result = pipe.analyze_text("I feel overwhelmed by exams this week.")
    assert result.analysis.wellbeing_shadow is None
    assert resolve_wellbeing_shadow_enabled() is False


def test_pipeline_compare_attaches_comparison_without_changing_fta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WELLBEING_SHADOW_ENABLED", "1")
    monkeypatch.setenv("WELLBEING_AUTHORITY_MODE", "compare")
    fta = _fta("moderate_concern")
    block = AnalysisBlock(
        overall=_ev(),
        modalities=ModalityBundle(),
        final_temporal_assessment=fta,
    )
    before = fta.model_dump()
    fake_shadow = WellbeingShadowAnalysis(
        status="ok",
        policy_candidate=_candidate(indicator="high_concern"),
    )
    pipe = MyUniSentimentPipeline()
    with patch("src.pipeline.build_wellbeing_shadow", return_value=fake_shadow):
        out = pipe._with_wellbeing_shadow(block, transcript="structured-only")
    assert out.final_temporal_assessment is not None
    assert out.final_temporal_assessment.model_dump() == before
    assert out.wellbeing_shadow is not None
    assert out.wellbeing_shadow.authority_comparison is not None
    assert out.wellbeing_shadow.authority_comparison.mode == "compare"
    assert out.wellbeing_shadow.authority_comparison.affects_final_assessment is False
    assert out.wellbeing_shadow.authority_comparison.agreement == "different"


def test_video_scope_unsupported_modality() -> None:
    cmp = compare_wellbeing_policies(
        legacy_indicator="moderate_concern",
        policy_candidate=_candidate(),
        mode="compare",
        migration_scope="text",  # type: ignore[arg-type]
    )
    assert cmp.fallback_reason == "unsupported_modality"
    assert cmp.agreement == "candidate_unavailable"


def test_legacy_insufficient_candidate_available_code() -> None:
    cmp = compare_wellbeing_policies(
        legacy_indicator="insufficient_evidence",
        policy_candidate=_candidate(indicator="moderate_concern"),
        mode="compare",
    )
    assert "legacy_insufficient_candidate_available" in cmp.comparison_reason_codes
