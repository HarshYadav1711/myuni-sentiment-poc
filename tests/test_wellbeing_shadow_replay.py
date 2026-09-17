"""Phase 4C.4 end-to-end shadow chain replay tests (no ML / no video)."""

from __future__ import annotations

import pytest

from evaluation.wellbeing.shadow_replay import (
    assert_no_raw_text_keys,
    find_replay_case,
    replay_shadow_chain,
    run_shadow_replay,
    write_report_artifact,
)
from evaluation.wellbeing.shadow_replay_cases import (
    ALL_SHADOW_REPLAY_CASES,
    POLICY_VERSION,
    get_replay_cases,
    replay_counts_by_expected_indicator,
    replay_counts_by_modality,
)
from src.schemas import FinalTemporalAssessment
from src.wellbeing.schemas import WellbeingShadowAnalysis


def test_replay_corpus_size_and_coverage() -> None:
    assert len(ALL_SHADOW_REPLAY_CASES) >= 40
    by_mod = replay_counts_by_modality()
    by_ind = replay_counts_by_expected_indicator()
    assert by_mod.get("text", 0) >= 5
    assert by_mod.get("audio", 0) >= 4
    assert by_mod.get("caption", 0) >= 3
    assert by_mod.get("video", 0) >= 10
    assert by_mod.get("conflict", 0) + by_mod.get("failure", 0) >= 5
    assert by_ind.get("low_concern", 0) >= 3
    assert by_ind.get("moderate_concern", 0) >= 5
    assert by_ind.get("high_concern", 0) >= 3
    assert by_ind.get("insufficient_evidence", 0) >= 5


@pytest.mark.parametrize(
    "case_id",
    [c.case_id for c in ALL_SHADOW_REPLAY_CASES],
    ids=[c.case_id for c in ALL_SHADOW_REPLAY_CASES],
)
def test_each_shadow_replay_case(case_id: str) -> None:
    report = run_shadow_replay([find_replay_case(case_id)])
    result = report.case_results[0]
    assert result.passed, f"{case_id}: {result.failures}"


def test_full_shadow_replay_harness() -> None:
    report = run_shadow_replay()
    assert report.case_total == len(ALL_SHADOW_REPLAY_CASES)
    assert report.exact_agreement == report.case_total
    assert report.determinism_failures == 0
    assert report.input_mutation_failures == 0
    assert report.evidence_grounding_failures == 0
    assert report.privacy_key_failures == 0
    assert report.source_boundary_failures == 0
    assert report.authority_isolation_failures == 0
    assert report.chain_invariant_count == 12
    assert report.chain_invariant_violations == 0
    assert report.all_passed


def test_chain_invariants() -> None:
    report = run_shadow_replay()
    for inv in report.invariant_results:
        assert inv.passed, f"{inv.invariant_id} {inv.description}: {inv.violations}"


def test_controlled_video_replay() -> None:
    case = find_replay_case("video_controlled_phase4b_replay")
    evidence, policy = replay_shadow_chain(case)
    assert evidence.temporal_evidence.eligible_window_count == 1
    assert evidence.temporal_evidence.eligible_window_indices == [3]
    assert policy.indicator == "moderate_concern"
    assert policy.status == "ok"
    assert policy.affects_final_assessment is False
    assert policy.policy_version == POLICY_VERSION


def test_structural_regressions_a_e_i_j() -> None:
    mapping = {
        "regression_A_personal_self_distress": "moderate_concern",
        "regression_E_personal_self_recovery": "low_concern",
        "regression_I_roommate_conflict_uncertain": "insufficient_evidence",
        "regression_J_personal_recovery": "low_concern",
    }
    for case_id, indicator in mapping.items():
        case = find_replay_case(case_id)
        assert case.structural_only is True
        _, policy = replay_shadow_chain(case)
        assert policy.indicator == indicator


def test_quoted_other_topic_figurative_regressions() -> None:
    for case_id in (
        "regression_quoted_other_blocked",
        "regression_topic_only",
        "regression_figurative_non_personal",
    ):
        _, policy = replay_shadow_chain(find_replay_case(case_id))
        assert policy.indicator == "insufficient_evidence"


def test_privacy_keys_recursive() -> None:
    case = find_replay_case("video_controlled_phase4b_replay")
    evidence, policy = replay_shadow_chain(case)
    shadow = WellbeingShadowAnalysis(
        status="ok",
        source_results=list(case.source_results),
        window_results=list(case.window_results),
        evidence_context=evidence,
        policy_candidate=policy,
    )
    for payload in (shadow.model_dump(), evidence.model_dump(), policy.model_dump()):
        assert assert_no_raw_text_keys(payload) == []


def test_authority_isolation_final_temporal_assessment() -> None:
    fields = set(FinalTemporalAssessment.model_fields.keys())
    assert "policy_candidate" not in fields
    assert "evidence_context" not in fields
    assert "wellbeing_shadow" not in fields

    # Mocked existing assessment object remains independent of shadow candidate.
    fta = FinalTemporalAssessment(
        overall_wellbeing_indicator="moderate_concern",
    )
    case = find_replay_case("video_persistent_local_distress")
    _, policy = replay_shadow_chain(case)
    assert policy.indicator == "high_concern"
    assert policy.affects_final_assessment is False
    # FTA object not mutated / not replaced by shadow policy.
    assert fta.overall_wellbeing_indicator == "moderate_concern"


def test_policy_version_stability() -> None:
    _, policy = replay_shadow_chain(find_replay_case("text_personal_distress"))
    assert policy.policy_version == "phase4c2-v1"


def test_expectations_independent_of_builders() -> None:
    import inspect
    import evaluation.wellbeing.shadow_replay_cases as cases_mod

    src = inspect.getsource(cases_mod)
    assert "from src.wellbeing.evidence" not in src
    assert "from src.wellbeing.policy_candidate" not in src
    assert "import src.wellbeing.evidence" not in src
    assert "import src.wellbeing.policy_candidate" not in src
    assert "ReplayExpectation(" in src
    assert "ALL_SHADOW_REPLAY_CASES" in src


def test_modality_filters() -> None:
    videos = get_replay_cases(modality="video")
    assert videos
    assert all(c.modality == "video" for c in videos)
    assert get_replay_cases(tag="replay")


def test_report_artifact_writable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    report = run_shadow_replay()
    path = write_report_artifact(report, path=tmp_path / "_phase4c4_report.json")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "exact_agreement" in text
    assert "phase4c2-v1" in text


def test_no_visual_ocr_sources_in_corpus() -> None:
    for case in ALL_SHADOW_REPLAY_CASES:
        for src in case.source_results:
            assert src.source_role not in {"ocr", "visual", "siglip", "face"}
