"""Phase 4C.3 — deterministic wellbeing policy validation harness.

Engineering validation of phase4c2-v1 against structured evidence fixtures.

NOT clinical accuracy / sensitivity / specificity / medical validation.
No ML inference. No network. No user text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from evaluation.wellbeing.policy_scenarios import (
    ALL_POLICY_SCENARIOS,
    POLICY_VERSION_UNDER_TEST,
    PolicyScenario,
    get_policy_scenarios,
    scenario_counts_by_branch,
    scenario_counts_by_indicator,
    scenario_counts_by_status,
)
from src.wellbeing.labels import DISTRESS_LIKE_SIGNAL_IDS
from src.wellbeing.policy_candidate import build_wellbeing_policy_candidate
from src.wellbeing.schemas import (
    WellbeingEvidenceContext,
    WellbeingGlobalEvidence,
    WellbeingPolicyCandidate,
    WellbeingSignalTemporalEvidence,
    WellbeingTemporalEvidence,
)

VALID_INDICATORS = frozenset(
    {"low_concern", "moderate_concern", "high_concern", "insufficient_evidence"},
)
VALID_STATUSES = frozenset(
    {"ok", "insufficient_evidence", "conflict", "unavailable"},
)

INVARIANT_IDS = tuple(f"I{i}" for i in range(1, 11))


@dataclass
class ScenarioResult:
    scenario_id: str
    passed: bool
    indicator_match: bool
    status_match: bool
    pattern_match: bool
    reason_code_match: bool
    actual_indicator: str
    actual_status: str
    expected_indicator: str
    expected_status: str
    failures: list[str] = field(default_factory=list)


@dataclass
class InvariantResult:
    invariant_id: str
    description: str
    passed: bool
    violations: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Engineering metrics for Phase 4C.3 (not clinical metrics)."""

    scenario_total: int = 0
    exact_indicator_agreement: int = 0
    exact_status_agreement: int = 0
    reason_code_expectation_agreement: int = 0
    pattern_agreement: int = 0
    invariant_count: int = 0
    invariant_violations: int = 0
    determinism_failures: int = 0
    evidence_grounding_failures: int = 0
    scenario_results: list[ScenarioResult] = field(default_factory=list)
    invariant_results: list[InvariantResult] = field(default_factory=list)
    counts_by_expected_indicator: dict[str, int] = field(default_factory=dict)
    counts_by_expected_status: dict[str, int] = field(default_factory=dict)
    counts_by_branch: dict[str, int] = field(default_factory=dict)
    matrix_checks_run: int = 0
    matrix_failures: list[str] = field(default_factory=list)

    @property
    def all_scenarios_passed(self) -> bool:
        return all(r.passed for r in self.scenario_results) and self.invariant_violations == 0

    def summary_dict(self) -> dict[str, Any]:
        return {
            "scenario_total": self.scenario_total,
            "exact_indicator_agreement": self.exact_indicator_agreement,
            "exact_status_agreement": self.exact_status_agreement,
            "reason_code_expectation_agreement": self.reason_code_expectation_agreement,
            "pattern_agreement": self.pattern_agreement,
            "invariant_count": self.invariant_count,
            "invariant_violations": self.invariant_violations,
            "determinism_failures": self.determinism_failures,
            "evidence_grounding_failures": self.evidence_grounding_failures,
            "counts_by_expected_indicator": dict(self.counts_by_expected_indicator),
            "counts_by_expected_status": dict(self.counts_by_expected_status),
            "counts_by_branch": dict(self.counts_by_branch),
            "matrix_checks_run": self.matrix_checks_run,
            "all_scenarios_passed": self.all_scenarios_passed,
            "policy_version": POLICY_VERSION_UNDER_TEST,
        }


def _evaluate_scenario(scenario: PolicyScenario) -> ScenarioResult:
    expected = scenario.expected
    # Deep-copy evidence so mutations would be detectable.
    evidence = None if scenario.evidence is None else scenario.evidence.model_copy(deep=True)
    before = None if evidence is None else evidence.model_dump()
    actual = build_wellbeing_policy_candidate(evidence)
    after = None if evidence is None else evidence.model_dump()

    failures: list[str] = []
    if before != after:
        failures.append("input_evidence_mutated")

    indicator_match = actual.indicator == expected.indicator
    status_match = actual.status == expected.status
    pattern_match = (
        actual.local_support_level == expected.local_support_level
        and actual.distress_pattern == expected.distress_pattern
        and actual.recovery_pattern == expected.recovery_pattern
    )
    if not indicator_match:
        failures.append(
            f"indicator expected={expected.indicator} actual={actual.indicator}",
        )
    if not status_match:
        failures.append(f"status expected={expected.status} actual={actual.status}")
    if actual.local_support_level != expected.local_support_level:
        failures.append(
            f"local_support expected={expected.local_support_level} "
            f"actual={actual.local_support_level}",
        )
    if actual.distress_pattern != expected.distress_pattern:
        failures.append(
            f"distress_pattern expected={expected.distress_pattern} "
            f"actual={actual.distress_pattern}",
        )
    if actual.recovery_pattern != expected.recovery_pattern:
        failures.append(
            f"recovery_pattern expected={expected.recovery_pattern} "
            f"actual={actual.recovery_pattern}",
        )

    missing_reasons = [
        code for code in expected.required_reason_codes if code not in actual.reason_codes
    ]
    forbidden_hit = [
        code for code in expected.forbidden_reason_codes if code in actual.reason_codes
    ]
    reason_code_match = not missing_reasons and not forbidden_hit
    if missing_reasons:
        failures.append(f"missing_reason_codes={missing_reasons}")
    if forbidden_hit:
        failures.append(f"forbidden_reason_codes_present={forbidden_hit}")

    if actual.policy_version != POLICY_VERSION_UNDER_TEST:
        failures.append(f"policy_version={actual.policy_version}")
    if actual.affects_final_assessment is not False:
        failures.append("affects_final_assessment_not_false")
    if actual.indicator not in VALID_INDICATORS:
        failures.append(f"invalid_indicator={actual.indicator}")
    if actual.status not in VALID_STATUSES:
        failures.append(f"invalid_status={actual.status}")
    if len(actual.reason_codes) != len(set(actual.reason_codes)):
        failures.append("duplicate_reason_codes")

    # Evidence grounding
    if evidence is not None:
        eid_set = set(evidence.evidence_ids)
        for sid in actual.supporting_evidence_ids:
            if sid not in eid_set:
                failures.append(f"ungrounded_supporting_id={sid}")
        for cid in actual.conflict_evidence_ids:
            if cid not in eid_set:
                failures.append(f"ungrounded_conflict_id={cid}")

    # Determinism
    again = build_wellbeing_policy_candidate(
        None if scenario.evidence is None else scenario.evidence.model_copy(deep=True),
    )
    if again.model_dump() != actual.model_dump():
        failures.append("determinism_failure")

    # No raw-text fields
    blob = str(actual.model_dump()).lower()
    for bad in ("transcript_text", "raw_text", "speech_segments", "utterance"):
        if bad in blob:
            failures.append(f"raw_text_field_leak={bad}")

    passed = not failures
    return ScenarioResult(
        scenario_id=scenario.scenario_id,
        passed=passed,
        indicator_match=indicator_match,
        status_match=status_match,
        pattern_match=pattern_match,
        reason_code_match=reason_code_match,
        actual_indicator=actual.indicator,
        actual_status=actual.status,
        expected_indicator=expected.indicator,
        expected_status=expected.status,
        failures=failures,
    )


def _global_eligible(evidence: Optional[WellbeingEvidenceContext]) -> bool:
    if evidence is None:
        return False
    g = evidence.global_evidence
    return bool(g.source_present and g.eligibility_status == "eligible")


def _global_has_distress(evidence: Optional[WellbeingEvidenceContext]) -> bool:
    if evidence is None:
        return False
    return any(s in DISTRESS_LIKE_SIGNAL_IDS for s in evidence.global_evidence.selected_signals)


def _distress_window_indices(evidence: WellbeingEvidenceContext) -> list[int]:
    idxs: set[int] = set()
    for sig in evidence.temporal_evidence.signal_evidence:
        if sig.signal in DISTRESS_LIKE_SIGNAL_IDS:
            idxs.update(sig.window_indices)
    return sorted(idxs)


def _has_consecutive_pair(indices: Sequence[int]) -> bool:
    ordered = sorted(set(indices))
    return any(ordered[i] + 1 == ordered[i + 1] for i in range(len(ordered) - 1))


def check_invariants(
    scenarios: Sequence[PolicyScenario],
) -> list[InvariantResult]:
    """Validate explicit safety invariants against scenario outcomes."""
    results: list[InvariantResult] = []

    # I1: no globally eligible → never low/moderate/high
    v1: list[str] = []
    for sc in scenarios:
        out = build_wellbeing_policy_candidate(sc.evidence)
        if not _global_eligible(sc.evidence) and out.indicator in {
            "low_concern",
            "moderate_concern",
            "high_concern",
        }:
            v1.append(sc.scenario_id)
    results.append(
        InvariantResult(
            "I1",
            "No globally eligible personal wellbeing → never low/moderate/high",
            passed=not v1,
            violations=v1,
        ),
    )

    # I2: high requires global eligible distress AND >=2 consecutive distress windows
    v2: list[str] = []
    for sc in scenarios:
        out = build_wellbeing_policy_candidate(sc.evidence)
        if out.indicator != "high_concern":
            continue
        if sc.evidence is None or not _global_eligible(sc.evidence):
            v2.append(sc.scenario_id)
            continue
        if not _global_has_distress(sc.evidence):
            v2.append(sc.scenario_id)
            continue
        idxs = _distress_window_indices(sc.evidence)
        if not _has_consecutive_pair(idxs):
            v2.append(sc.scenario_id)
    results.append(
        InvariantResult(
            "I2",
            "High requires global eligible distress + >=2 consecutive distress windows",
            passed=not v2,
            violations=v2,
        ),
    )

    # I3–I5: isolated named signals must not produce high
    for inv_id, tag, signal in (
        ("I3", "invariant3", "hopelessness_like_language"),
        ("I4", "invariant4", "self_directed_negativity"),
        ("I5", "invariant5", "anxiety_or_fear_language"),
    ):
        v: list[str] = []
        for sc in scenarios:
            if tag not in sc.tags and signal not in sc.scenario_id:
                # Still check isolated single-window cases named for the signal
                pass
            out = build_wellbeing_policy_candidate(sc.evidence)
            if sc.evidence is None:
                continue
            idxs = _distress_window_indices(sc.evidence)
            # Exactly one distress window and that window's only selected distress is the signal
            # Use scenario tags for targeted invariants.
            if tag in sc.tags and out.indicator == "high_concern":
                v.append(sc.scenario_id)
        results.append(
            InvariantResult(
                inv_id,
                f"Isolated {signal} must not force high",
                passed=not v,
                violations=v,
            ),
        )

    # I6: different distress labels across adjacent windows may form persistent support
    v6: list[str] = []
    for sc in scenarios:
        if "invariant6" not in sc.tags:
            continue
        out = build_wellbeing_policy_candidate(sc.evidence)
        if out.indicator != "high_concern":
            v6.append(sc.scenario_id)
    results.append(
        InvariantResult(
            "I6",
            "Different adjacent distress labels may form persistent temporal support",
            passed=not v6,
            violations=v6,
        ),
    )

    # I7: recovery does not simply subtract from distress (conflict/low rules preserved)
    v7: list[str] = []
    for sc in scenarios:
        if "invariant7" not in sc.tags:
            continue
        out = build_wellbeing_policy_candidate(sc.evidence)
        # Must not invent a fake averaged middle category field
        blob = str(out.model_dump()).lower()
        if "averaged" in blob or "net_distress" in blob:
            v7.append(sc.scenario_id)
        # Tagged recovery+distress scenarios should match human expectations already;
        # invariant: never high when global recovery-only with isolated local distress
        if sc.scenario_id == "low_global_recovery_isolated_local_distress":
            if out.indicator != "low_concern":
                v7.append(sc.scenario_id)
        if sc.scenario_id == "conflict_global_recovery_persistent_local":
            if out.status != "conflict":
                v7.append(sc.scenario_id)
    results.append(
        InvariantResult(
            "I7",
            "Recovery does not simply subtract from distress",
            passed=not v7,
            violations=v7,
        ),
    )

    # I8: material mixed conflict never averaged into fake middle
    v8: list[str] = []
    for sc in scenarios:
        if "invariant8" not in sc.tags and "conflict" not in sc.tags:
            continue
        if sc.expected.status != "conflict":
            continue
        out = build_wellbeing_policy_candidate(sc.evidence)
        if out.status != "conflict" or out.indicator != "insufficient_evidence":
            v8.append(sc.scenario_id)
        if out.indicator in {"low_concern", "moderate_concern", "high_concern"}:
            v8.append(sc.scenario_id)
    results.append(
        InvariantResult(
            "I8",
            "Material mixed recovery/distress conflict never averaged into category",
            passed=not v8,
            violations=v8,
        ),
    )

    # I9: local cannot override globally non-personal attribution
    v9: list[str] = []
    for sc in scenarios:
        if "invariant9" not in sc.tags:
            continue
        out = build_wellbeing_policy_candidate(sc.evidence)
        if out.indicator in {"low_concern", "moderate_concern", "high_concern"}:
            v9.append(sc.scenario_id)
    results.append(
        InvariantResult(
            "I9",
            "Local windows cannot override globally non-personal attribution",
            passed=not v9,
            violations=v9,
        ),
    )

    # I10: global personal distress usable when local lacks context
    v10: list[str] = []
    for sc in scenarios:
        if "invariant10" not in sc.tags:
            continue
        out = build_wellbeing_policy_candidate(sc.evidence)
        if out.indicator != "moderate_concern":
            v10.append(sc.scenario_id)
    results.append(
        InvariantResult(
            "I10",
            "Global personal distress remains usable with zero/weak local support",
            passed=not v10,
            violations=v10,
        ),
    )

    return results


def _matrix_evidence_variants() -> list[WellbeingEvidenceContext]:
    """Bounded programmatic matrix of evidence states (not thousands)."""
    variants: list[WellbeingEvidenceContext] = []
    global_opts = [
        WellbeingGlobalEvidence(
            source_present=True,
            source_role="transcript",
            classification_status="ok",
            eligibility_status="eligible",
            personal_wellbeing_eligible=True,
            selected_signals=["stress_or_overwhelm"],
        ),
        WellbeingGlobalEvidence(
            source_present=True,
            source_role="transcript",
            classification_status="ok",
            eligibility_status="eligible",
            personal_wellbeing_eligible=True,
            selected_signals=["positive_wellbeing_or_recovery"],
        ),
        WellbeingGlobalEvidence(
            source_present=True,
            source_role="transcript",
            classification_status="ok",
            eligibility_status="not_eligible",
            personal_wellbeing_eligible=False,
            selected_signals=[],
        ),
        WellbeingGlobalEvidence(
            source_present=True,
            source_role="transcript",
            classification_status="ok",
            eligibility_status="uncertain",
            personal_wellbeing_eligible=False,
            selected_signals=[],
        ),
        WellbeingGlobalEvidence(source_present=False),
    ]
    temporal_opts = [
        WellbeingTemporalEvidence(),
        WellbeingTemporalEvidence(
            evaluated_window_count=1,
            eligible_window_count=1,
            eligible_window_indices=[0],
            eligible_window_fraction=1.0,
            distress_window_count=1,
            signal_evidence=[
                WellbeingSignalTemporalEvidence(
                    signal="stress_or_overwhelm",
                    window_count=1,
                    window_indices=[0],
                    first_start=0.0,
                    last_end=5.0,
                ),
            ],
        ),
        WellbeingTemporalEvidence(
            evaluated_window_count=2,
            eligible_window_count=2,
            eligible_window_indices=[0, 1],
            eligible_window_fraction=1.0,
            longest_eligible_run_windows=2,
            longest_eligible_run_seconds=10.0,
            distress_window_count=2,
            signal_evidence=[
                WellbeingSignalTemporalEvidence(
                    signal="stress_or_overwhelm",
                    window_count=1,
                    window_indices=[0],
                    first_start=0.0,
                    last_end=5.0,
                ),
                WellbeingSignalTemporalEvidence(
                    signal="academic_pressure",
                    window_count=1,
                    window_indices=[1],
                    first_start=5.0,
                    last_end=10.0,
                ),
            ],
        ),
        WellbeingTemporalEvidence(
            evaluated_window_count=3,
            eligible_window_count=2,
            eligible_window_indices=[0, 2],
            eligible_window_fraction=2 / 3,
            distress_window_count=2,
            signal_evidence=[
                WellbeingSignalTemporalEvidence(
                    signal="stress_or_overwhelm",
                    window_count=2,
                    window_indices=[0, 2],
                    first_start=0.0,
                    last_end=15.0,
                ),
            ],
        ),
    ]
    for g in global_opts:
        for t in temporal_opts:
            eids: list[str] = []
            if g.source_present and g.source_role:
                eids.append(f"wellbeing-global-{g.source_role}")
            for idx in t.eligible_window_indices:
                eids.append(f"wellbeing-window-{idx}")
            variants.append(
                WellbeingEvidenceContext(
                    status="ok" if g.source_present else "insufficient",
                    global_evidence=g,
                    temporal_evidence=t,
                    evidence_ids=eids,
                ),
            )
    return variants


def run_matrix_checks() -> tuple[int, list[str]]:
    """Property-like structural checks over a bounded evidence matrix."""
    failures: list[str] = []
    variants = _matrix_evidence_variants()
    for i, evidence in enumerate(variants):
        original = evidence.model_dump()
        a = build_wellbeing_policy_candidate(evidence)
        b = build_wellbeing_policy_candidate(evidence)
        if evidence.model_dump() != original:
            failures.append(f"matrix[{i}] mutated_input")
        if a.model_dump() != b.model_dump():
            failures.append(f"matrix[{i}] nondeterministic")
        if a.indicator not in VALID_INDICATORS:
            failures.append(f"matrix[{i}] bad_indicator={a.indicator}")
        if a.status not in VALID_STATUSES:
            failures.append(f"matrix[{i}] bad_status={a.status}")
        if a.policy_version != POLICY_VERSION_UNDER_TEST:
            failures.append(f"matrix[{i}] bad_version")
        if a.affects_final_assessment is not False:
            failures.append(f"matrix[{i}] affects_final")
        if len(a.reason_codes) != len(set(a.reason_codes)):
            failures.append(f"matrix[{i}] duplicate_reasons")
        eid_set = set(evidence.evidence_ids)
        for sid in a.supporting_evidence_ids + a.conflict_evidence_ids:
            if sid not in eid_set:
                failures.append(f"matrix[{i}] ungrounded_id={sid}")
        # Enum structural: high requires consecutive when evidence has indices
        if a.indicator == "high_concern":
            if not _global_eligible(evidence) or not _global_has_distress(evidence):
                failures.append(f"matrix[{i}] high_without_global_distress")
            idxs = _distress_window_indices(evidence)
            if not _has_consecutive_pair(idxs):
                failures.append(f"matrix[{i}] high_without_consecutive")
    return len(variants), failures


def run_policy_validation(
    scenarios: Optional[Sequence[PolicyScenario]] = None,
) -> ValidationReport:
    """Run full structured validation and return engineering metrics."""
    scenarios = list(scenarios) if scenarios is not None else list(ALL_POLICY_SCENARIOS)
    report = ValidationReport(
        scenario_total=len(scenarios),
        counts_by_expected_indicator=scenario_counts_by_indicator(),
        counts_by_expected_status=scenario_counts_by_status(),
        counts_by_branch=scenario_counts_by_branch(),
    )

    for sc in scenarios:
        result = _evaluate_scenario(sc)
        report.scenario_results.append(result)
        if result.indicator_match:
            report.exact_indicator_agreement += 1
        if result.status_match:
            report.exact_status_agreement += 1
        if result.reason_code_match:
            report.reason_code_expectation_agreement += 1
        if result.pattern_match:
            report.pattern_agreement += 1
        if any(f.startswith("determinism") for f in result.failures):
            report.determinism_failures += 1
        if any("ungrounded" in f for f in result.failures):
            report.evidence_grounding_failures += 1

    inv = check_invariants(scenarios)
    report.invariant_results = inv
    report.invariant_count = len(inv)
    report.invariant_violations = sum(1 for r in inv if not r.passed)

    matrix_n, matrix_failures = run_matrix_checks()
    report.matrix_checks_run = matrix_n
    report.matrix_failures = matrix_failures
    if matrix_failures:
        report.determinism_failures += sum(
            1 for f in matrix_failures if "nondeterministic" in f or "mutated" in f
        )
        report.evidence_grounding_failures += sum(
            1 for f in matrix_failures if "ungrounded" in f
        )

    return report


def find_scenario(scenario_id: str) -> PolicyScenario:
    for sc in ALL_POLICY_SCENARIOS:
        if sc.scenario_id == scenario_id:
            return sc
    raise KeyError(scenario_id)


__all__ = [
    "ValidationReport",
    "check_invariants",
    "find_scenario",
    "get_policy_scenarios",
    "run_matrix_checks",
    "run_policy_validation",
]
