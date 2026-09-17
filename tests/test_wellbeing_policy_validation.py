"""Phase 4C.3 structured wellbeing policy validation tests (no ML / no video)."""

from __future__ import annotations

import pytest

from evaluation.wellbeing.policy_scenarios import (
    ALL_POLICY_SCENARIOS,
    POLICY_VERSION_UNDER_TEST,
    get_policy_scenarios,
    scenario_counts_by_branch,
    scenario_counts_by_indicator,
    scenario_counts_by_status,
)
from evaluation.wellbeing.policy_validation import (
    find_scenario,
    run_matrix_checks,
    run_policy_validation,
)
from src.wellbeing.policy_candidate import build_wellbeing_policy_candidate
from src.schemas import FinalTemporalAssessment


def test_scenario_corpus_size_and_branch_coverage() -> None:
    assert len(ALL_POLICY_SCENARIOS) >= 60
    by_ind = scenario_counts_by_indicator()
    by_status = scenario_counts_by_status()
    by_branch = scenario_counts_by_branch()
    assert by_ind.get("low_concern", 0) >= 5
    assert by_ind.get("moderate_concern", 0) >= 5
    assert by_ind.get("high_concern", 0) >= 5
    assert by_ind.get("insufficient_evidence", 0) >= 5
    assert by_status.get("ok", 0) >= 5
    assert by_status.get("conflict", 0) >= 3
    assert by_branch.get("insufficient", 0) >= 5
    assert by_branch.get("low", 0) >= 5
    assert by_branch.get("moderate", 0) >= 5
    assert by_branch.get("high", 0) >= 5
    assert by_branch.get("conflict", 0) >= 3


@pytest.mark.parametrize(
    "scenario_id",
    [s.scenario_id for s in ALL_POLICY_SCENARIOS],
    ids=[s.scenario_id for s in ALL_POLICY_SCENARIOS],
)
def test_each_structured_scenario(scenario_id: str) -> None:
    report = run_policy_validation([find_scenario(scenario_id)])
    assert report.scenario_total == 1
    result = report.scenario_results[0]
    assert result.passed, f"{scenario_id}: {result.failures}"


def test_full_validation_harness_passes() -> None:
    report = run_policy_validation()
    assert report.scenario_total == len(ALL_POLICY_SCENARIOS)
    assert report.exact_indicator_agreement == report.scenario_total
    assert report.exact_status_agreement == report.scenario_total
    assert report.reason_code_expectation_agreement == report.scenario_total
    assert report.pattern_agreement == report.scenario_total
    assert report.invariant_count == 10
    assert report.invariant_violations == 0
    assert report.determinism_failures == 0
    assert report.evidence_grounding_failures == 0
    assert report.matrix_checks_run >= 10
    assert report.matrix_failures == []
    assert report.all_scenarios_passed


def test_invariants_all_pass() -> None:
    report = run_policy_validation()
    for inv in report.invariant_results:
        assert inv.passed, f"{inv.invariant_id} {inv.description}: {inv.violations}"


def test_controlled_video_structured_replay() -> None:
    sc = find_scenario("replay_controlled_video_phase4b")
    out = build_wellbeing_policy_candidate(sc.evidence)
    assert out.indicator == "moderate_concern"
    assert out.status == "ok"
    assert out.policy_version == POLICY_VERSION_UNDER_TEST
    assert out.affects_final_assessment is False
    # Isolated hopelessness/self-negativity in window 3 must not force high.
    assert out.indicator != "high_concern"


def test_isolated_hopelessness_not_high() -> None:
    out = build_wellbeing_policy_candidate(
        find_scenario("mod_isolated_hopelessness").evidence,
    )
    assert out.indicator == "moderate_concern"


def test_isolated_self_negativity_not_high() -> None:
    out = build_wellbeing_policy_candidate(
        find_scenario("mod_isolated_self_negativity").evidence,
    )
    assert out.indicator == "moderate_concern"


def test_persistent_mixed_distress_labels_high() -> None:
    out = build_wellbeing_policy_candidate(
        find_scenario("high_mixed_labels_consecutive").evidence,
    )
    assert out.indicator == "high_concern"


def test_global_not_eligible_persistent_local_insufficient() -> None:
    out = build_wellbeing_policy_candidate(
        find_scenario("insuf_local_only_persistent").evidence,
    )
    assert out.indicator == "insufficient_evidence"
    assert "local_eligible_without_global_eligibility" in out.reason_codes


def test_global_recovery_persistent_local_distress_conflict() -> None:
    out = build_wellbeing_policy_candidate(
        find_scenario("conflict_global_recovery_persistent_local").evidence,
    )
    assert out.status == "conflict"
    assert out.indicator == "insufficient_evidence"


def test_monotonicity_ladder() -> None:
    """Escalation comes from temporal recurrence, not signal name."""
    zero = build_wellbeing_policy_candidate(
        find_scenario("mod_global_distress_zero_local").evidence,
    )
    one = build_wellbeing_policy_candidate(
        find_scenario("mod_global_distress_one_local").evidence,
    )
    sep = build_wellbeing_policy_candidate(
        find_scenario("mod_global_distress_separated").evidence,
    )
    two = build_wellbeing_policy_candidate(
        find_scenario("high_two_consecutive").evidence,
    )
    assert zero.indicator == "moderate_concern"
    assert one.indicator == "moderate_concern"
    assert sep.indicator == "moderate_concern"
    assert two.indicator == "high_concern"
    # Signal-name-only change does not escalate.
    renamed = build_wellbeing_policy_candidate(
        find_scenario("mod_signal_name_change_no_escalate").evidence,
    )
    assert renamed.indicator == "moderate_concern"


def test_recovery_semantics_documented() -> None:
    """phase4c2-v1 recovery semantics (engineering, not clinical)."""
    only = build_wellbeing_policy_candidate(
        find_scenario("low_global_recovery_only").evidence,
    )
    windows = build_wellbeing_policy_candidate(
        find_scenario("low_repeated_local_recovery").evidence,
    )
    isolated_distress = build_wellbeing_policy_candidate(
        find_scenario("low_global_recovery_isolated_local_distress").evidence,
    )
    persistent = build_wellbeing_policy_candidate(
        find_scenario("conflict_global_recovery_persistent_local").evidence,
    )
    assert only.indicator == "low_concern"
    assert windows.indicator == "low_concern"
    assert isolated_distress.indicator == "low_concern"  # isolated local distress allowed
    assert persistent.status == "conflict"
    assert persistent.indicator == "insufficient_evidence"


def test_window_index_adjacency_not_timestamps() -> None:
    consecutive = build_wellbeing_policy_candidate(
        find_scenario("high_indices_1_2_consecutive").evidence,
    )
    nonconsec = build_wellbeing_policy_candidate(
        find_scenario("mod_indices_1_3_nonconsecutive").evidence,
    )
    short = build_wellbeing_policy_candidate(
        find_scenario("high_short_final_window").evidence,
    )
    assert consecutive.indicator == "high_concern"
    assert nonconsec.indicator == "moderate_concern"
    assert short.indicator == "high_concern"


def test_adversarial_scenarios_fail_safe() -> None:
    for sid in (
        "mod_all_eight_distress_one_window",
        "mod_twenty_nonconsecutive_distress",
        "insuf_local_only_persistent",
        "conflict_global_recovery_persistent_local",
        "adversarial_missing_evidence_ids",
        "insuf_classifier_error_populated_fields",
        "mod_out_of_order_window_construction",
    ):
        sc = find_scenario(sid)
        out = build_wellbeing_policy_candidate(sc.evidence)
        assert out.indicator in {
            "low_concern",
            "moderate_concern",
            "high_concern",
            "insufficient_evidence",
        }
        assert out.affects_final_assessment is False
        assert out.policy_version == POLICY_VERSION_UNDER_TEST


def test_matrix_property_checks() -> None:
    n, failures = run_matrix_checks()
    assert n >= 10
    assert failures == []


def test_expectations_independent_of_policy_module_source() -> None:
    """Fixtures must not import/call policy when defining expected values."""
    import inspect
    import evaluation.wellbeing.policy_scenarios as scenarios_mod

    src = inspect.getsource(scenarios_mod)
    assert "from src.wellbeing.policy_candidate" not in src
    assert "import src.wellbeing.policy_candidate" not in src
    # Expectations are PolicyExpectation literals — not policy return values.
    assert "PolicyExpectation(" in src
    assert "ALL_POLICY_SCENARIOS" in src


def test_no_authority_wiring() -> None:
    fields = set(FinalTemporalAssessment.model_fields.keys())
    assert "policy_candidate" not in fields
    assert "wellbeing_policy" not in fields


def test_filter_helpers() -> None:
    lows = get_policy_scenarios(branch="low")
    assert lows
    assert all(s.expected.indicator == "low_concern" for s in lows)
    replay = get_policy_scenarios(tag="replay")
    assert len(replay) == 1
    assert replay[0].scenario_id == "replay_controlled_video_phase4b"


def test_decision_coverage_reported() -> None:
    report = run_policy_validation()
    # Every major indicator/status branch exercised by expected corpus.
    for key in ("low_concern", "moderate_concern", "high_concern", "insufficient_evidence"):
        assert report.counts_by_expected_indicator.get(key, 0) > 0
    for key in ("ok", "conflict", "insufficient_evidence", "unavailable"):
        assert report.counts_by_expected_status.get(key, 0) > 0


def test_input_not_mutated_across_full_run() -> None:
    snapshots = [
        (None if s.evidence is None else s.evidence.model_dump())
        for s in ALL_POLICY_SCENARIOS
    ]
    run_policy_validation()
    for sc, before in zip(ALL_POLICY_SCENARIOS, snapshots):
        after = None if sc.evidence is None else sc.evidence.model_dump()
        assert after == before
